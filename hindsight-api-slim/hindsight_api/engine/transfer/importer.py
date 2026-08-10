"""Import documents from a transfer archive by replaying the deterministic retain pipeline.

For each document the importer rebuilds the extracted facts, re-embeds them with
the *target* bank's embedding model, then runs entity resolution (Phase 1) and
the fact/link insert (Phase 2) — exactly the steps retain runs after LLM
extraction. No LLM is called. Temporal/semantic/causal links and entity merges
are therefore computed relative to the target bank's existing memories.
"""

from __future__ import annotations

import io
import json
import logging
import uuid
import zipfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any, Literal

from ..causal_links import CANONICAL_CAUSAL_LINK_TYPE, LEGACY_CAUSAL_LINK_TYPES
from ..db_utils import acquire_with_retry
from ..retain import bank_utils, chunk_storage, embedding_processing, fact_storage, link_utils, orchestrator
from ..retain.types import (
    CausalRelation,
    ChunkMetadata,
    ExtractedFact,
    ProcessedFact,
    RetainContent,
)
from ..schema import fq_table
from .schema import (
    CARRIED_HISTORY_TABLES,
    HISTORY_TABLES,
    SCHEMA_VERSION,
    BankRowsJSONEncoding,
    TransferDocument,
    TransferFact,
    TransferManifest,
    TransferObservation,
)

logger = logging.getLogger(__name__)

OnConflict = Literal["skip", "replace", "new-id"]
_VALID_CONFLICT_MODES: tuple[OnConflict, ...] = ("skip", "replace", "new-id")


@dataclass
class ImportedDocument:
    """A single document successfully imported, with the units it produced.

    Carried back so the engine can fire the post-retain extension hook
    (usage tracking / metrics / notifications) once per imported document,
    mirroring how retain reports each completed document.
    """

    document_id: str
    unit_ids: list[str]
    content: str
    tags: list[str]


@dataclass
class ImportResult:
    """Outcome of importing a transfer archive into a bank."""

    documents_imported: int = 0
    documents_skipped: int = 0
    facts_imported: int = 0
    observations_imported: int = 0
    # Observations dropped because some source fact was not imported in this run.
    observations_skipped: int = 0
    skipped_document_ids: list[str] = field(default_factory=list)
    # Original id -> freshly generated id, for documents imported under "new-id".
    remapped_document_ids: dict[str, str] = field(default_factory=dict)
    # Per-document outcomes, for the engine's post-retain hook. Not serialized
    # into operation result_metadata (the worker handler writes counts only).
    imported_documents: list[ImportedDocument] = field(default_factory=list)


@dataclass
class _ObservationOutcome:
    """Counts from the observation import pass."""

    imported: int = 0
    skipped: int = 0


@dataclass
class _ImportedFactBatch:
    """Inserted fact IDs paired with their ordinals in the source archive."""

    unit_ids: list[str]
    original_ordinals: list[int]


@dataclass
class ParsedArchive:
    """A transfer archive after parsing/validation."""

    manifest: TransferManifest
    documents: list[TransferDocument]
    observations: list[TransferObservation] = field(default_factory=list)


def parse_archive(archive_bytes: bytes) -> ParsedArchive:
    """Parse and validate a transfer ZIP archive produced by ``export_documents``."""
    with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as zf:
        names = set(zf.namelist())
        if "manifest.json" not in names:
            raise ValueError("Invalid transfer archive: manifest.json is missing")
        manifest = TransferManifest.model_validate_json(zf.read("manifest.json"))
        if manifest.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported transfer archive schema version {manifest.schema_version} "
                f"(this build supports {SCHEMA_VERSION})"
            )
        doc_names = sorted(n for n in names if n.startswith("documents/") and n.endswith(".json"))
        documents = [TransferDocument.model_validate_json(zf.read(name)) for name in doc_names]
        observations: list[TransferObservation] = []
        if "observations.json" in names:
            observations = [TransferObservation.model_validate(o) for o in json.loads(zf.read("observations.json"))]
    return ParsedArchive(manifest=manifest, documents=documents, observations=observations)


async def import_documents(
    *,
    backend: Any,
    embeddings_model: Any,
    entity_resolver: Any,
    config: Any,
    format_date_fn: Any,
    bank_id: str,
    archive_bytes: bytes,
    on_conflict: OnConflict = "skip",
    ops: Any = None,
    outbox_callback_factory: Any = None,
) -> ImportResult:
    """Import every document in ``archive_bytes`` into ``bank_id``.

    Args:
        backend: Database backend (provides ``acquire()`` and ``ops``).
        embeddings_model: Target bank's embedding model (used to re-embed facts).
        entity_resolver: Shared entity resolver for the target bank.
        config: Resolved bank config for the target bank.
        format_date_fn: Date formatter used when augmenting fact text for embedding
            (must match retain so embeddings are consistent).
        bank_id: Target bank.
        archive_bytes: A ZIP archive produced by ``export_documents``.
        on_conflict: How to handle a document id that already exists in the target
            bank — ``skip`` (default), ``replace`` (delete old data and re-import),
            or ``new-id`` (import under a freshly generated id).
        ops: Backend ``DataAccessOps``. Defaults to ``backend.ops``.

    Returns:
        An :class:`ImportResult` with per-document counts.
    """
    if on_conflict not in _VALID_CONFLICT_MODES:
        raise ValueError(f"Invalid on_conflict '{on_conflict}'; expected one of {_VALID_CONFLICT_MODES}")
    if ops is None:
        ops = backend.ops

    parsed = parse_archive(archive_bytes)
    result = ImportResult()

    # (original document_id, fact ordinal) -> freshly inserted unit id. Used to
    # resolve observation source references after all facts exist.
    ref_map: dict[tuple[str, int], str] = {}

    for document in parsed.documents:
        target_id = await _resolve_target_id(backend, bank_id, document.id, on_conflict)
        if target_id is None:
            result.documents_skipped += 1
            result.skipped_document_ids.append(document.id)
            continue
        if target_id != document.id:
            result.remapped_document_ids[document.id] = target_id

        imported_facts = await _import_one_document(
            backend=backend,
            embeddings_model=embeddings_model,
            entity_resolver=entity_resolver,
            config=config,
            format_date_fn=format_date_fn,
            bank_id=bank_id,
            document=document,
            target_id=target_id,
            ops=ops,
            outbox_callback_factory=outbox_callback_factory,
        )
        result.documents_imported += 1
        result.facts_imported += len(imported_facts.unit_ids)
        result.imported_documents.append(
            ImportedDocument(
                document_id=target_id,
                unit_ids=imported_facts.unit_ids,
                content=document.original_text or "",
                tags=list(document.tags),
            )
        )
        for ordinal, unit_id in zip(imported_facts.original_ordinals, imported_facts.unit_ids, strict=True):
            ref_map[(document.id, ordinal)] = unit_id

    if parsed.observations:
        outcome = await _import_observations(
            backend=backend,
            embeddings_model=embeddings_model,
            bank_id=bank_id,
            observations=parsed.observations,
            ref_map=ref_map,
            ops=ops,
        )
        result.observations_imported = outcome.imported
        result.observations_skipped = outcome.skipped

    logger.info(
        "[transfer] Imported %d document(s), %d fact(s), %d observation(s) into bank %s "
        "(%d docs skipped, %d observations skipped)",
        result.documents_imported,
        result.facts_imported,
        result.observations_imported,
        bank_id,
        result.documents_skipped,
        result.observations_skipped,
    )
    return result


# Bank-level config/state tables restored from a whole-bank archive. Order matters
# for foreign keys: banks precede every child; mental_models precede their
# Knowledge Pages; Knowledge Page parents precede children.
_BANK_CHILD_TABLES = ("mental_models", "knowledge_pages", "directives", "webhooks")
# Child-history carried verbatim; restored after its parent (mental_models) so the
# foreign key resolves. Surrogate ids were dropped on export (the target reassigns
# them), so these restore via fresh IDENTITY values.


@dataclass
class BankImportResult:
    """Outcome of importing a whole-bank archive."""

    bank_id: str
    documents_imported: int = 0
    facts_imported: int = 0
    observations_imported: int = 0
    mental_models_imported: int = 0
    knowledge_pages_imported: int = 0
    mental_model_history_imported: int = 0
    directives_imported: int = 0
    webhooks_imported: int = 0
    history_rows_imported: int = 0


@dataclass
class ParsedBankArchive:
    """The bank-level sections of a whole-bank archive (documents read separately)."""

    manifest: TransferManifest
    # table name -> carried row dicts (banks, mental_models, Knowledge Pages,
    # directives, webhooks)
    bank_rows: dict[str, list[dict]] = field(default_factory=dict)
    # table name -> rows (audit_log, llm_requests), present only with --include-history
    history_rows: dict[str, list[dict]] = field(default_factory=dict)


def parse_bank_archive(archive_bytes: bytes) -> ParsedBankArchive:
    """Parse the bank-level sections of a whole-bank archive (``archive_type='bank'``)."""
    with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as zf:
        names = set(zf.namelist())
        if "manifest.json" not in names:
            raise ValueError("Invalid transfer archive: manifest.json is missing")
        manifest = TransferManifest.model_validate_json(zf.read("manifest.json"))
        if manifest.archive_type != "bank":
            raise ValueError(
                f"Not a whole-bank archive (archive_type={manifest.archive_type!r}); use import_documents instead"
            )
        bank_rows: dict[str, list[dict]] = {}
        for table in ("banks", *_BANK_CHILD_TABLES, *CARRIED_HISTORY_TABLES):
            fname = f"{table}.json"
            bank_rows[table] = json.loads(zf.read(fname)) if fname in names else []
        history_rows: dict[str, list[dict]] = {}
        for table in HISTORY_TABLES:
            fname = f"history/{table}.json"
            if fname in names:
                history_rows[table] = json.loads(zf.read(fname))
    return ParsedBankArchive(manifest=manifest, bank_rows=bank_rows, history_rows=history_rows)


def _resolve_bank_rows_json_encoding(manifest: TransferManifest) -> BankRowsJSONEncoding:
    """Resolve row JSON provenance, including the released v1 archive contract."""
    return manifest.bank_rows_json_encoding or "decoded"


async def _restore_rows(
    conn: Any,
    table: str,
    rows: list[dict],
    *,
    bank_rows_json_encoding: BankRowsJSONEncoding = "decoded",
) -> int:
    """Insert verbatim rows into a bank-scoped table, coercing JSON-encoded values
    back to the column's type (timestamps, uuids, jsonb). ``ON CONFLICT DO NOTHING``
    keeps an import idempotent and safe to re-run against a partially-filled target."""
    if not rows:
        return 0
    from ..memory_engine import get_current_schema

    schema = get_current_schema()
    col_types = {
        r["column_name"]: r["data_type"]
        for r in await conn.fetch(
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = $1 AND table_name = $2",
            schema,
            table,
        )
    }
    inserted = 0
    for row in rows:
        cols = [c for c in row if c in col_types]
        placeholders: list[str] = []
        values: list[Any] = []
        for position, col in enumerate(cols, start=1):
            data_type = col_types[col]
            value = row[col]
            if data_type in ("jsonb", "json"):
                # asyncpg has no JSON codec on these raw connections; pass JSON
                # text and cast. Provenance is required because a decoded JSON
                # scalar containing JSON text is indistinguishable from a raw
                # serialized object after the outer archive JSON is parsed.
                if value is not None and (bank_rows_json_encoding == "decoded" or not isinstance(value, str)):
                    value = json.dumps(value)
                values.append(value)
                placeholders.append(f"${position}::jsonb")
                continue
            if value is not None and isinstance(value, str):
                if data_type in ("timestamp with time zone", "timestamp without time zone"):
                    value = datetime.fromisoformat(value)
                elif data_type == "date":
                    value = date.fromisoformat(value)
                elif data_type == "uuid":
                    value = uuid.UUID(value)
            placeholders.append(f"${position}")
            values.append(value)
        col_list = ", ".join(f'"{c}"' for c in cols)
        await conn.execute(
            f"INSERT INTO {fq_table(table)} ({col_list}) VALUES ({', '.join(placeholders)}) ON CONFLICT DO NOTHING",
            *values,
        )
        inserted += 1
    return inserted


def _knowledge_pages_parent_first(rows: list[dict]) -> list[dict]:
    """Return a stable parent-before-child ordering for a Knowledge Page tree.

    The table has a self-referential ``parent_id`` foreign key, so archive order
    is not a valid restore order. Reject dangling parents and cycles before any
    import writes occur instead of leaving a partially restored tree.
    """
    if not rows:
        return []

    row_ids = [row.get("id") for row in rows]
    if any(row_id is None for row_id in row_ids):
        raise ValueError("Invalid Knowledge Pages archive: every node must have an id")
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("Invalid Knowledge Pages archive: duplicate node id")

    known_ids = set(row_ids)
    missing_parents = {
        row.get("parent_id")
        for row in rows
        if row.get("parent_id") is not None and row.get("parent_id") not in known_ids
    }
    if missing_parents:
        raise ValueError(f"Invalid Knowledge Pages archive: missing parent node(s): {sorted(missing_parents)!r}")

    ordered: list[dict] = []
    restored_ids: set[Any] = set()
    pending = list(rows)
    while pending:
        ready = [row for row in pending if row.get("parent_id") is None or row.get("parent_id") in restored_ids]
        if not ready:
            unresolved = sorted(str(row["id"]) for row in pending)
            raise ValueError(f"Invalid Knowledge Pages archive: parent cycle among node(s): {unresolved!r}")
        for row in ready:
            ordered.append(row)
            restored_ids.add(row["id"])
        ready_ids = {id(row) for row in ready}
        pending = [row for row in pending if id(row) not in ready_ids]
    return ordered


def _validate_knowledge_page_models(rows: list[dict], mental_model_rows: list[dict]) -> None:
    """Reject Knowledge Page nodes that cannot preserve their source meaning.

    A folder is structural and must not reference a mental model. A page is a
    projection of exactly one carried mental model, so accepting a missing or
    dangling reference would either fail after document replay or silently
    restore a page with no content.
    """
    mental_model_ids = {row.get("id") for row in mental_model_rows if row.get("id") is not None}
    nodes_by_id = {row.get("id"): row for row in rows}
    for row in rows:
        node_id = row.get("id")
        kind = row.get("kind")
        mental_model_id = row.get("mental_model_id")
        if kind == "folder":
            if mental_model_id is not None:
                raise ValueError(f"Invalid Knowledge Pages archive: folder '{node_id}' references a mental model")
        elif kind == "page":
            if mental_model_id is None or mental_model_id not in mental_model_ids:
                raise ValueError(
                    f"Invalid Knowledge Pages archive: page '{node_id}' references missing mental model "
                    f"'{mental_model_id}'"
                )
        else:
            raise ValueError(f"Invalid Knowledge Pages archive: node '{node_id}' has unsupported kind '{kind}'")

        parent_id = row.get("parent_id")
        if parent_id is not None and nodes_by_id[parent_id].get("kind") != "folder":
            raise ValueError(f"Invalid Knowledge Pages archive: parent '{parent_id}' is not a folder")


async def _reembed_mental_model_rows(rows: list[dict], embeddings_model: Any) -> list[dict]:
    """Attach target-model embeddings to carried mental-model rows.

    Export deliberately strips encoder-specific vectors. Generate every target
    vector before the import writes its bank row so an embedding failure cannot
    leave a half-created target bank.
    """
    if not rows:
        return []
    texts = [f"{row.get('name') or ''} {row.get('content') or ''}" for row in rows]
    embeddings = await embedding_processing.generate_embeddings_batch(embeddings_model, texts)
    if len(embeddings) != len(rows):
        raise RuntimeError(
            f"Mental-model embedding count mismatch during bank import: expected {len(rows)}, got {len(embeddings)}"
        )
    return [{**row, "embedding": str(embedding)} for row, embedding in zip(rows, embeddings)]


async def _restore_mental_models(
    conn: Any,
    rows: list[dict],
    *,
    bank_id: str,
    config: Any,
    ops: Any,
    bank_rows_json_encoding: BankRowsJSONEncoding,
) -> int:
    """Restore carried mental models and rebuild the target text projection."""
    inserted = await _restore_rows(
        conn,
        "mental_models",
        rows,
        bank_rows_json_encoding=bank_rows_json_encoding,
    )
    # Native PostgreSQL regenerates its generated tsvector on INSERT, while
    # base-column backends index name/content directly. VChord is the only
    # backend needing an application-maintained mental-model projection.
    search_vector_expr = ops.mental_model_search_vector_expr(config)
    if inserted and search_vector_expr:
        await conn.execute(
            f"UPDATE {fq_table('mental_models')} SET search_vector = {search_vector_expr} WHERE bank_id = $1",
            bank_id,
        )
    return inserted


async def import_bank(
    *,
    backend: Any,
    embeddings_model: Any,
    entity_resolver: Any,
    resolve_config: Callable[[], Awaitable[Any]],
    format_date_fn: Any,
    archive_bytes: bytes,
    target_bank_id: str | None = None,
    include_history: bool = False,
    ops: Any = None,
) -> BankImportResult:
    """Restore a whole bank from a ``export_bank`` archive into the target instance.

    Re-embeds facts and mental models with the *target* instance's embedding model,
    rebuilds their search projections, and rebuilds links/entities — the path for
    migrating a bank to an instance configured with a different embedding model /
    vector / text-search backend. Knowledge Page ids, hierarchy and backing
    mental-model references are preserved.

    The **target bank must not already exist**: import restores a complete bank
    (config + facts + mental models + …) and is not a merge. If a bank with the
    target id is present, this raises — delete it first or pass ``target_bank_id``
    for a fresh id. A migration restores *exact* state, so unlike the document
    import it fires no retain webhooks and triggers no consolidation/graph
    maintenance: observations and mental models are restored as exported.

    Takes ``resolve_config`` rather than a resolved config because the only correct
    moment to resolve one is *inside* this function, after the archive's bank row
    lands. Before that the target bank does not exist (import refuses to write into
    an existing one), so any config a caller resolved carries global + tenant values
    and none of the bank's own — which is exactly the bug in #3236.
    """
    if ops is None:
        ops = backend.ops
    parsed = parse_bank_archive(archive_bytes)
    bank_rows_json_encoding = _resolve_bank_rows_json_encoding(parsed.manifest)
    source_bank_id = parsed.manifest.source_bank_id
    bank_id = target_bank_id or source_bank_id

    # Remapping to a different id: rewrite the carried bank_id on every row so FKs
    # and PKs line up with the (also-remapped) documents/facts.
    if bank_id != source_bank_id:
        for rows in (*parsed.bank_rows.values(), *parsed.history_rows.values()):
            for row in rows:
                if "bank_id" in row:
                    row["bank_id"] = bank_id

    # Validate all Knowledge Page relationships before the first write. These can
    # fail independently of PostgreSQL and should leave no partially-created
    # target bank.
    knowledge_page_rows = _knowledge_pages_parent_first(parsed.bank_rows.get("knowledge_pages", []))
    raw_mental_model_rows = parsed.bank_rows.get("mental_models", [])
    _validate_knowledge_page_models(knowledge_page_rows, raw_mental_model_rows)

    # Fail an obvious existing-target import before paying to embed mental
    # models. The guarded check immediately before INSERT remains below so two
    # concurrent importers cannot race this preflight into a merge.
    async with acquire_with_retry(backend) as conn:
        if await conn.fetchval(f"SELECT 1 FROM {fq_table('banks')} WHERE bank_id = $1", bank_id):
            raise ValueError(
                f"Target bank '{bank_id}' already exists; import-bank restores into a fresh bank "
                f"(it is not a merge). Delete the bank first, or pass a different target bank id."
            )

    # Generate every target-model vector before creating the bank, so an
    # embedding-provider failure cannot leave a partially restored target.
    mental_model_rows = await _reembed_mental_model_rows(
        raw_mental_model_rows,
        embeddings_model,
    )

    async with acquire_with_retry(backend) as conn:
        # Refuse to import into an existing bank — this restores a whole bank, it
        # does not merge. Merging would silently mix the archive's config/mental
        # models/webhooks with whatever is already there (and global-unique ids
        # like webhooks/directives would collide).
        if await conn.fetchval(f"SELECT 1 FROM {fq_table('banks')} WHERE bank_id = $1", bank_id):
            raise ValueError(
                f"Target bank '{bank_id}' already exists; import-bank restores into a fresh bank "
                f"(it is not a merge). Delete the bank first, or pass a different target bank id."
            )
        # Bank row first — children (documents, mental_models, …) FK to it.
        await _restore_rows(
            conn,
            "banks",
            parsed.bank_rows.get("banks", []),
            bank_rows_json_encoding=bank_rows_json_encoding,
        )
        # The restored banks row bypasses the fresh-INSERT gate that normally
        # creates per-bank vector indexes, so create them explicitly here while
        # the bank is still empty (facts are imported below, so the build is
        # instant). get_or_create_bank_profile would NOT do this: the row now
        # exists, so it takes the SELECT branch and skips index creation —
        # leaving the restored bank falling back to the global index +
        # post-filter (slower, under-returning recall). See #2645.
        internal_id = await conn.fetchval(f"SELECT internal_id FROM {fq_table('banks')} WHERE bank_id = $1", bank_id)
        if internal_id is not None:
            await bank_utils.create_bank_vector_indexes(conn, bank_id, str(internal_id), ops=ops)

    # Only now does the bank row — and with it the archive's own config — exist, so
    # this is where the config the documents are replayed with has to come from.
    # Until #3236 the import ran on a config resolved before the restore, which
    # could not contain the bank's `entity_labels`: every label entity was
    # classified as a regular one, which both exposed label values to fuzzy merging
    # (#3187) and left them inside the trigram index that the partial index is
    # supposed to keep them out of (#3208), so an imported bank silently lost that
    # fix.
    config = await resolve_config()

    doc_result = await import_documents(
        backend=backend,
        embeddings_model=embeddings_model,
        entity_resolver=entity_resolver,
        config=config,
        format_date_fn=format_date_fn,
        bank_id=bank_id,
        archive_bytes=archive_bytes,
        ops=ops,
        outbox_callback_factory=None,
    )

    result = BankImportResult(
        bank_id=bank_id,
        documents_imported=doc_result.documents_imported,
        facts_imported=doc_result.facts_imported,
        observations_imported=doc_result.observations_imported,
    )
    async with acquire_with_retry(backend) as conn:
        result.mental_models_imported = await _restore_mental_models(
            conn,
            mental_model_rows,
            bank_id=bank_id,
            config=config,
            ops=ops,
            bank_rows_json_encoding=bank_rows_json_encoding,
        )
        result.knowledge_pages_imported = await _restore_rows(
            conn,
            "knowledge_pages",
            knowledge_page_rows,
            bank_rows_json_encoding=bank_rows_json_encoding,
        )
        # Restored after mental_models so the (mental_model_id, bank_id) FK resolves.
        result.mental_model_history_imported = await _restore_rows(
            conn,
            "mental_model_history",
            parsed.bank_rows.get("mental_model_history", []),
            bank_rows_json_encoding=bank_rows_json_encoding,
        )
        result.directives_imported = await _restore_rows(
            conn,
            "directives",
            parsed.bank_rows.get("directives", []),
            bank_rows_json_encoding=bank_rows_json_encoding,
        )
        result.webhooks_imported = await _restore_rows(
            conn,
            "webhooks",
            parsed.bank_rows.get("webhooks", []),
            bank_rows_json_encoding=bank_rows_json_encoding,
        )
        if include_history:
            for table in HISTORY_TABLES:
                result.history_rows_imported += await _restore_rows(
                    conn,
                    table,
                    parsed.history_rows.get(table, []),
                    bank_rows_json_encoding=bank_rows_json_encoding,
                )

    logger.info(
        "[transfer] Imported bank %s: %d doc(s), %d fact(s), %d observation(s), "
        "%d mental model(s), %d Knowledge Page node(s), %d mm-history row(s), "
        "%d directive(s), %d webhook(s), %d history row(s)",
        bank_id,
        result.documents_imported,
        result.facts_imported,
        result.observations_imported,
        result.mental_models_imported,
        result.knowledge_pages_imported,
        result.mental_model_history_imported,
        result.directives_imported,
        result.webhooks_imported,
        result.history_rows_imported,
    )
    return result


async def _resolve_target_id(backend: Any, bank_id: str, document_id: str, on_conflict: OnConflict) -> str | None:
    """Decide the document id to write under, or ``None`` to skip.

    Returns the original id when there is no conflict, a fresh id under
    ``new-id``, the original id under ``replace`` (the insert path cascades the
    old data away), or ``None`` under ``skip`` when the document already exists.
    """
    async with acquire_with_retry(backend) as conn:
        exists = await conn.fetchval(
            f"SELECT 1 FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2",
            document_id,
            bank_id,
        )
    if not exists:
        return document_id
    if on_conflict == "skip":
        return None
    if on_conflict == "new-id":
        return str(uuid.uuid4())
    return document_id  # replace


async def _import_one_document(
    *,
    backend: Any,
    embeddings_model: Any,
    entity_resolver: Any,
    config: Any,
    format_date_fn: Any,
    bank_id: str,
    document: TransferDocument,
    target_id: str,
    ops: Any,
    outbox_callback_factory: Any = None,
) -> _ImportedFactBatch:
    """Re-embed and insert a document; map original fact ordinals to new unit ids."""
    log_buffer: list[str] = []

    # Fire the same retain.completed webhook retain emits, transactionally inside
    # this document's insert. Factory returns None when no webhook manager exists.
    outbox_callback = (
        outbox_callback_factory([{"document_id": target_id, "tags": list(document.tags)}])
        if outbox_callback_factory
        else None
    )

    extracted_facts = [_to_extracted_fact(fact) for fact in document.facts]
    legacy_causal_relations = _legacy_causal_relations(document)

    processed_facts: list[ProcessedFact] = []
    retained_index_by_original: list[int | None] = []
    if extracted_facts:
        augmented = embedding_processing.augment_texts_with_dates(extracted_facts, format_date_fn)
        embeddings = await embedding_processing.generate_embeddings_batch(embeddings_model, augmented)
        fact_batch = orchestrator._process_extracted_facts(extracted_facts, embeddings)
        extracted_facts = fact_batch.extracted_facts
        processed_facts = fact_batch.processed_facts
        retained_index_by_original = fact_batch.retained_index_by_original
        legacy_causal_relations = orchestrator._remap_causal_relations(
            legacy_causal_relations,
            retained_index_by_original,
        )

    contents = [RetainContent(content=document.original_text or "")]
    chunk_meta = [
        ChunkMetadata(chunk_text=chunk.chunk_text, fact_count=0, content_index=0, chunk_index=chunk.chunk_index)
        for chunk in document.chunks
    ]

    # Phase 1 (entity resolution + semantic ANN) on its own connection, outside
    # the write transaction — mirrors the retain pipeline.
    entity_resolver.discard_pending_stats()
    phase1 = await orchestrator._pre_resolve_phase1(
        backend,
        entity_resolver,
        bank_id,
        contents,
        processed_facts,
        config,
        log_buffer,
        skip_semantic_ann=False,
    )

    async with acquire_with_retry(backend) as conn:
        async with conn.transaction():
            # is_first_batch=True: cascade-delete any existing data for this id
            # (the "replace" path) and (re)insert the document row.
            await fact_storage.handle_document_tracking(
                conn,
                bank_id,
                target_id,
                document.original_text or "",
                True,
                document.retain_params,
                document.tags,
                ops=ops,
            )
            if document.created_at is not None:
                # Transfer archives carry source provenance. Apply it here,
                # without changing normal retain/upsert timestamp semantics.
                await conn.execute(
                    f"UPDATE {fq_table('documents')} SET created_at = $1 WHERE id = $2 AND bank_id = $3",
                    document.created_at,
                    target_id,
                    bank_id,
                )

            chunk_id_map: dict[int, str] = {}
            if chunk_meta:
                chunk_id_map = await chunk_storage.store_chunks_batch(conn, bank_id, target_id, chunk_meta, ops=ops)

            for extracted, processed in zip(extracted_facts, processed_facts):
                processed.document_id = target_id
                if chunk_id_map and extracted.chunk_index is not None:
                    chunk_id = chunk_id_map.get(extracted.chunk_index)
                    if chunk_id:
                        processed.chunk_id = chunk_id

            result_unit_ids = await orchestrator._insert_facts_and_links(
                conn,
                entity_resolver,
                bank_id,
                contents,
                extracted_facts,
                processed_facts,
                config,
                log_buffer,
                resolved_entities=phase1.entities.resolved_entities,
                entity_to_unit=phase1.entities.entity_to_unit,
                unit_to_entity_ids=phase1.entities.unit_to_entity_ids,
                semantic_ann_links=phase1.semantic_ann_links,
                skip_semantic_links=False,
                outbox_callback=outbox_callback,
                ops=ops,
            )

            # Retain writes only ``caused_by``. Restore legacy archive edges
            # separately so their distinct direction and semantics survive a
            # transfer without broadening the normal retain write contract.
            if result_unit_ids and legacy_causal_relations:
                await link_utils.restore_legacy_causal_links_batch(
                    conn,
                    bank_id,
                    result_unit_ids[0],
                    legacy_causal_relations,
                    ops=ops,
                )

            # Restore the source consolidation lifecycle. A whole-bank transfer
            # preserves exact eligibility: a fact that was consolidated (or that
            # failed consolidation) in the source is never re-consolidated on the
            # target, so the maintenance reconciler sees no phantom backlog and
            # observations are not re-derived. Archives predating these fields
            # carry None for all three -> skipped here, leaving the
            # observation-driven marking in _import_observations as the only
            # (lossy) signal, exactly as before.
            if result_unit_ids:
                await _restore_fact_lifecycle(
                    conn,
                    bank_id,
                    document.facts,
                    retained_index_by_original,
                    result_unit_ids[0],
                )

    # Best-effort, and only after the acquire() block above has exited: this
    # takes its own connection, and on Oracle the write above is not committed
    # until that block exits, so flushing while still holding the connection
    # deadlocks (see the retain orchestrator for the full explanation).
    try:
        await entity_resolver.flush_pending_stats()
    except Exception:
        logger.warning("[transfer] Entity stats flush failed for document %s", target_id, exc_info=True)

    logger.debug("[transfer] Imported document %s:\n%s", target_id, "\n".join(log_buffer))
    # Single content item -> result_unit_ids[0] follows the retained fact order.
    retained_unit_ids = list(result_unit_ids[0]) if result_unit_ids else []
    return _ImportedFactBatch(
        unit_ids=retained_unit_ids,
        original_ordinals=[
            original_index
            for original_index, retained_index in enumerate(retained_index_by_original)
            if retained_index is not None
        ],
    )


async def _restore_fact_lifecycle(
    conn: Any,
    bank_id: str,
    facts: list[TransferFact],
    retained_index_by_original: list[int | None],
    retained_unit_ids: list[str],
) -> None:
    """Apply each imported fact's source consolidation timestamps to its new row.

    ``retained_unit_ids`` follows the retained fact order; ``retained_index_by_original[i]``
    maps original fact ``i`` to its position there (or ``None`` if it was dropped
    on insert, e.g. a duplicate). ``created_at`` restores source provenance only
    when present (mirroring the document-row handling); ``consolidated_at`` /
    ``consolidation_failed_at`` are set verbatim — a source-``NULL`` (unconsolidated)
    fact stays eligible, which is correct.
    """
    rows: list[tuple[uuid.UUID, datetime | None, datetime | None, datetime | None]] = []
    for original_index, fact in enumerate(facts):
        retained_index = retained_index_by_original[original_index]
        if retained_index is None:
            continue
        if fact.created_at is None and fact.consolidated_at is None and fact.consolidation_failed_at is None:
            # Legacy archive without lifecycle fields — nothing to restore.
            continue
        rows.append(
            (
                uuid.UUID(retained_unit_ids[retained_index]),
                fact.created_at,
                fact.consolidated_at,
                fact.consolidation_failed_at,
            )
        )
    if not rows:
        return
    await conn.executemany(
        f"UPDATE {fq_table('memory_units')} "
        f"SET created_at = COALESCE($2, created_at), consolidated_at = $3, consolidation_failed_at = $4 "
        f"WHERE id = $1 AND bank_id = $5",
        [
            (unit_id, created_at, consolidated_at, failed_at, bank_id)
            for unit_id, created_at, consolidated_at, failed_at in rows
        ],
    )


async def _import_observations(
    *,
    backend: Any,
    embeddings_model: Any,
    bank_id: str,
    observations: list[TransferObservation],
    ref_map: dict[tuple[str, int], str],
    ops: Any,
) -> _ObservationOutcome:
    """Insert observations whose source facts were all imported in this run.

    Observations carry no embedding, links, or entity rows — only the unit row
    plus ``source_memory_ids`` (remapped to the freshly inserted source units)
    and ``proof_count``. Their source facts are marked ``consolidated_at`` so the
    target bank's consolidator won't re-process them. Mirrors what consolidation
    writes, but driven from the archive instead of the LLM.

    Inserted as-is: imported observations are NOT merged or deduplicated against
    observations that already exist in the target bank (unlike consolidation,
    which merges related observations). Importing into a bank that already has
    observations — or importing the same archive twice — can therefore produce
    overlapping observations over the same facts.
    """
    outcome = _ObservationOutcome()

    # Resolve each observation's sources to new unit ids; drop any whose sources
    # weren't all imported (e.g. a subset/skip import).
    resolved: list[tuple[TransferObservation, list[str]]] = []
    for obs in observations:
        source_ids = [ref_map.get((s.document_id, s.fact_index)) for s in obs.sources]
        if not source_ids or any(sid is None for sid in source_ids):
            outcome.skipped += 1
            continue
        resolved.append((obs, [sid for sid in source_ids if sid is not None]))

    if not resolved:
        return outcome

    # Observations embed the raw text (matching consolidation), not the
    # date-augmented text used for facts.
    embeddings = await embedding_processing.generate_embeddings_batch(
        embeddings_model, [obs.text for obs, _ in resolved]
    )
    processed = [
        ProcessedFact(
            fact_text=obs.text,
            fact_type="observation",
            embedding=embedding,
            occurred_start=obs.occurred_start,
            occurred_end=obs.occurred_end,
            mentioned_at=_observation_mentioned_at(obs),
            context="",
            metadata={},
            tags=list(obs.tags),
            observation_scopes=obs.observation_scopes,
            document_id=None,
            chunk_id=None,
        )
        for (obs, _sources), embedding in zip(resolved, embeddings)
    ]

    async with acquire_with_retry(backend) as conn:
        async with conn.transaction():
            obs_unit_ids = await fact_storage.insert_facts_batch(conn, bank_id, processed, ops=ops)

            all_source_ids: set[uuid.UUID] = set()
            for (obs, sources), obs_unit_id in zip(resolved, obs_unit_ids):
                observation_uuid = uuid.UUID(obs_unit_id)
                if obs.event_date is not None:
                    # insert_facts_batch derives event_date for normal writes;
                    # transfer restores the source value carried by the archive.
                    await conn.execute(
                        f"UPDATE {fq_table('memory_units')} SET event_date = $1 WHERE id = $2 AND bank_id = $3",
                        obs.event_date,
                        observation_uuid,
                        bank_id,
                    )
                source_uuids = [uuid.UUID(s) for s in sources]
                all_source_ids.update(source_uuids)
                await _link_observation_sources(conn, ops, bank_id, observation_uuid, source_uuids, obs.proof_count)

            # Mark source facts consolidated so the target consolidator skips
            # them. COALESCE keeps the exact source timestamp already restored by
            # _restore_fact_lifecycle (new archives); now() is the fallback only
            # for legacy archives that carry no per-fact lifecycle state.
            if all_source_ids:
                await conn.execute(
                    f"UPDATE {fq_table('memory_units')} SET consolidated_at = COALESCE(consolidated_at, now()) "
                    f"WHERE bank_id = $1 AND id = ANY($2)",
                    bank_id,
                    list(all_source_ids),
                )

    outcome.imported = len(resolved)
    return outcome


async def _link_observation_sources(
    conn: Any,
    ops: Any,
    bank_id: str,
    observation_id: uuid.UUID,
    source_ids: list[uuid.UUID],
    proof_count: int,
) -> None:
    """Attach source ids + proof_count to a freshly inserted observation row.

    PG stores the sources in the ``source_memory_ids`` array column; Oracle uses
    the ``observation_sources`` junction table (same split as consolidation).
    """
    if ops.uses_observation_sources_table:
        await conn.executemany(
            f"INSERT INTO {fq_table('observation_sources')} (observation_id, source_id) "
            f"VALUES ($1, $2) ON CONFLICT (observation_id, source_id) DO NOTHING",
            [(observation_id, sid) for sid in dict.fromkeys(source_ids)],
        )
        await conn.execute(
            f"UPDATE {fq_table('memory_units')} SET proof_count = $1 WHERE id = $2 AND bank_id = $3",
            proof_count,
            observation_id,
            bank_id,
        )
    else:
        await conn.execute(
            f"UPDATE {fq_table('memory_units')} SET source_memory_ids = $1, proof_count = $2 "
            f"WHERE id = $3 AND bank_id = $4",
            source_ids,
            proof_count,
            observation_id,
            bank_id,
        )


def _observation_mentioned_at(obs: TransferObservation) -> datetime | None:
    """event_date (NOT NULL) is derived from occurred_start or mentioned_at on
    insert; fall back so the column stays populated for observations too."""
    mentioned_at = obs.mentioned_at
    if obs.occurred_start is None and mentioned_at is None:
        mentioned_at = obs.event_date or datetime.now(UTC)
    return mentioned_at


def _to_extracted_fact(fact: TransferFact) -> ExtractedFact:
    """Rebuild the retain pipeline's ExtractedFact from a serialized transfer fact."""
    # event_date is NOT NULL in the schema and is derived from occurred_start or
    # mentioned_at on insert. When neither is present, fall back to the carried
    # event_date (or now) via mentioned_at so the column stays populated.
    mentioned_at = fact.mentioned_at
    if fact.occurred_start is None and mentioned_at is None:
        mentioned_at = fact.event_date or datetime.now(UTC)

    return ExtractedFact(
        fact_text=fact.text,
        fact_type=fact.fact_type,
        entities=list(fact.entities),
        occurred_start=fact.occurred_start,
        occurred_end=fact.occurred_end,
        where=None,
        causal_relations=[
            CausalRelation(relation_type=rel.relation_type, target_fact_index=rel.target_fact_index)
            for rel in fact.causal_relations
            if rel.relation_type == CANONICAL_CAUSAL_LINK_TYPE
        ],
        content_index=0,
        chunk_index=fact.chunk_index,
        context=fact.context or "",
        mentioned_at=mentioned_at,
        metadata=dict(fact.metadata),
        tags=list(fact.tags),
        observation_scopes=fact.observation_scopes,
    )


def _legacy_causal_relations(document: TransferDocument) -> list[list[CausalRelation]]:
    """Return legacy archive edges for transfer-only restoration.

    Invalid archive values are excluded. The write helper repeats the explicit
    compatibility allowlist as a persistence boundary.
    """
    return [
        [
            CausalRelation(relation_type=relation.relation_type, target_fact_index=relation.target_fact_index)
            for relation in fact.causal_relations
            if relation.relation_type in LEGACY_CAUSAL_LINK_TYPES
        ]
        for fact in document.facts
    ]
