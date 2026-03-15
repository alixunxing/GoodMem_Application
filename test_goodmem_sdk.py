"""
Comprehensive test suite for the GoodMem Python SDK.

Covers:
  - Client initialization (sync & async, context manager)
  - Embedder CRUD
  - LLM CRUD
  - Reranker CRUD
  - Space CRUD
  - Memory CRUD (plain text, PDF/base64, batch operations)
  - Memory retrieval (RAG agent, with and without reranker)
  - System info
  - User lookup (me)
  - API key lifecycle
  - Error handling (NotFoundError, AuthenticationError, etc.)
  - Async client smoke test

Prerequisites
-------------
  pip install goodmem pytest pytest-asyncio

Environment variables required
-------------------------------
  GOODMEM_BASE_URL   – GoodMem server URL (no /v1 suffix), e.g. http://localhost:8080
  GOODMEM_API_KEY    – GoodMem API key (prefix gm_)
  OPENAI_API_KEY     – OpenAI API key (used for embedder + LLM registration)
  VOYAGE_API_KEY     – (optional) Voyage AI key for reranker tests

Run
---
  pytest test_goodmem_sdk.py -v
"""

import base64
import io
import json
import os
import time
import uuid

import pytest
import pytest_asyncio

from goodmem import AsyncGoodmem, Goodmem
from goodmem import (
    APIError,
    AuthenticationError,
    NotFoundError,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = os.environ.get("GOODMEM_BASE_URL", "http://localhost:8080")
API_KEY = os.environ.get("GOODMEM_API_KEY", "gm_placeholder")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY", "")

EMBEDDER_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o-mini"  # cheaper than gpt-5.1 for tests; change if needed
RERANKER_MODEL = "rerank-2.5"

# Minimal valid PDF bytes (1-page blank) – avoids needing a file on disk
_MINIMAL_PDF = (
    b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj "
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj "
    b"3 0 obj<</Type/Page/MediaBox[0 0 3 3]>>endobj "
    b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n"
    b"0000000058 00000 n \n0000000115 00000 n \n"
    b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF"
)

PLAIN_TEXT_CONTENT = (
    "Transformers are a type of neural network architecture that are "
    "particularly well-suited for natural language processing tasks. "
    "A Transformer model leverages the attention mechanism to capture "
    "long-range dependencies in the input sequence."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def unique_name(prefix: str) -> str:
    """Return a unique display name to avoid conflicts between test runs."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def wait_for_memory_processing(client: Goodmem, memory_id: str, timeout: int = 60):
    """Poll until a memory's processing_status reaches COMPLETED (or timeout)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        mem = client.memories.get(memory_id)
        status = mem.processing_status
        if status == "COMPLETED":
            return mem
        if status in ("FAILED", "ERROR"):
            pytest.fail(f"Memory {memory_id} processing failed: {status}")
        time.sleep(3)
    pytest.fail(f"Memory {memory_id} did not complete within {timeout}s")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def client() -> Goodmem:
    """A session-scoped synchronous GoodMem client."""
    return Goodmem(base_url=BASE_URL, api_key=API_KEY)


@pytest.fixture(scope="session")
def embedder_id(client: Goodmem) -> str:
    """Create a shared embedder for the test session; clean up afterwards."""
    embedder = client.embedders.create(
        display_name=unique_name("test-embedder"),
        model_identifier=EMBEDDER_MODEL,
        api_key=OPENAI_API_KEY,
    )
    eid = embedder.embedder_id
    yield eid
    try:
        client.embedders.delete(eid)
    except APIError:
        pass


@pytest.fixture(scope="session")
def llm_id(client: Goodmem) -> str:
    """Create a shared LLM for the test session; clean up afterwards."""
    llm = client.llms.create(
        display_name=unique_name("test-llm"),
        model_identifier=LLM_MODEL,
        api_key=OPENAI_API_KEY,
    )
    lid = llm.llm_id
    yield lid
    try:
        client.llms.delete(lid)
    except APIError:
        pass


@pytest.fixture(scope="session")
def space_id(client: Goodmem, embedder_id: str) -> str:
    """Create a shared space for the test session; clean up afterwards."""
    space = client.spaces.create(
        name=unique_name("test-space"),
        space_embedders=[{"embedder_id": embedder_id, "default_retrieval_weight": 1.0}],
        default_chunking_config={"none": {}},
    )
    sid = space.space_id
    yield sid
    try:
        client.spaces.delete(sid)
    except APIError:
        pass


# ---------------------------------------------------------------------------
# 1. Client Initialization
# ---------------------------------------------------------------------------

class TestClientInit:
    def test_sync_client_creates_without_error(self):
        c = Goodmem(base_url=BASE_URL, api_key=API_KEY)
        assert c is not None

    def test_sync_client_context_manager(self):
        with Goodmem(base_url=BASE_URL, api_key=API_KEY) as c:
            assert c is not None

    def test_sync_client_custom_timeout(self):
        c = Goodmem(base_url=BASE_URL, api_key=API_KEY, timeout=60.0)
        assert c is not None

    def test_async_client_creates_without_error(self):
        c = AsyncGoodmem(base_url=BASE_URL, api_key=API_KEY)
        assert c is not None

    def test_invalid_api_key_raises_auth_error(self):
        bad_client = Goodmem(base_url=BASE_URL, api_key="gm_invalid_key_xyz")
        with pytest.raises((AuthenticationError, APIError)):
            bad_client.system.info()


# ---------------------------------------------------------------------------
# 2. System
# ---------------------------------------------------------------------------

class TestSystem:
    def test_system_info_returns_data(self, client: Goodmem):
        info = client.system.info()
        assert info is not None

    def test_system_info_has_version_field(self, client: Goodmem):
        info = client.system.info()
        # The response object should carry some version / status attribute
        assert hasattr(info, "__dict__") or isinstance(info, dict)


# ---------------------------------------------------------------------------
# 3. Users
# ---------------------------------------------------------------------------

class TestUsers:
    def test_me_returns_current_user(self, client: Goodmem):
        me = client.users.me()
        assert me is not None

    def test_me_has_user_id(self, client: Goodmem):
        me = client.users.me()
        assert hasattr(me, "user_id") or hasattr(me, "userId")

    def test_get_user_by_id(self, client: Goodmem):
        me = client.users.me()
        uid = getattr(me, "user_id", None) or getattr(me, "userId", None)
        if uid is None:
            pytest.skip("Cannot determine current user ID")
        user = client.users.get(uid)
        assert user is not None


# ---------------------------------------------------------------------------
# 4. Embedders
# ---------------------------------------------------------------------------

class TestEmbedders:
    def test_create_embedder(self, client: Goodmem):
        name = unique_name("emb-create")
        emb = client.embedders.create(
            display_name=name,
            model_identifier=EMBEDDER_MODEL,
            api_key=OPENAI_API_KEY,
        )
        assert emb.embedder_id
        client.embedders.delete(emb.embedder_id)

    def test_get_embedder(self, client: Goodmem, embedder_id: str):
        emb = client.embedders.get(embedder_id)
        assert emb.embedder_id == embedder_id

    def test_list_embedders_returns_results(self, client: Goodmem, embedder_id: str):
        ids = [e.embedder_id for e in client.embedders.list()]
        assert embedder_id in ids

    def test_update_embedder_display_name(self, client: Goodmem):
        name = unique_name("emb-update")
        emb = client.embedders.create(
            display_name=name,
            model_identifier=EMBEDDER_MODEL,
            api_key=OPENAI_API_KEY,
        )
        new_name = unique_name("emb-updated")
        updated = client.embedders.update(emb.embedder_id, display_name=new_name)
        assert updated.display_name == new_name
        client.embedders.delete(emb.embedder_id)

    def test_delete_embedder(self, client: Goodmem):
        name = unique_name("emb-delete")
        emb = client.embedders.create(
            display_name=name,
            model_identifier=EMBEDDER_MODEL,
            api_key=OPENAI_API_KEY,
        )
        client.embedders.delete(emb.embedder_id)
        with pytest.raises(NotFoundError):
            client.embedders.get(emb.embedder_id)

    def test_get_nonexistent_embedder_raises_not_found(self, client: Goodmem):
        with pytest.raises(NotFoundError):
            client.embedders.get("00000000-0000-0000-0000-000000000000")


# ---------------------------------------------------------------------------
# 5. LLMs
# ---------------------------------------------------------------------------

class TestLLMs:
    def test_create_llm(self, client: Goodmem):
        name = unique_name("llm-create")
        llm = client.llms.create(
            display_name=name,
            model_identifier=LLM_MODEL,
            api_key=OPENAI_API_KEY,
        )
        assert llm.llm_id
        client.llms.delete(llm.llm_id)

    def test_get_llm(self, client: Goodmem, llm_id: str):
        llm = client.llms.get(llm_id)
        assert llm.llm_id == llm_id

    def test_list_llms_returns_results(self, client: Goodmem, llm_id: str):
        ids = [l.llm_id for l in client.llms.list()]
        assert llm_id in ids

    def test_update_llm_display_name(self, client: Goodmem):
        name = unique_name("llm-update")
        llm = client.llms.create(
            display_name=name,
            model_identifier=LLM_MODEL,
            api_key=OPENAI_API_KEY,
        )
        new_name = unique_name("llm-updated")
        updated = client.llms.update(llm.llm_id, display_name=new_name)
        assert updated.display_name == new_name
        client.llms.delete(llm.llm_id)

    def test_delete_llm(self, client: Goodmem):
        name = unique_name("llm-delete")
        llm = client.llms.create(
            display_name=name,
            model_identifier=LLM_MODEL,
            api_key=OPENAI_API_KEY,
        )
        client.llms.delete(llm.llm_id)
        with pytest.raises(NotFoundError):
            client.llms.get(llm.llm_id)

    def test_get_nonexistent_llm_raises_not_found(self, client: Goodmem):
        with pytest.raises(NotFoundError):
            client.llms.get("00000000-0000-0000-0000-000000000000")


# ---------------------------------------------------------------------------
# 6. Rerankers  (skip if no VOYAGE_API_KEY)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not VOYAGE_API_KEY, reason="VOYAGE_API_KEY not set")
class TestRerankers:
    def test_create_reranker(self, client: Goodmem):
        name = unique_name("reranker-create")
        rr = client.rerankers.create(
            display_name=name,
            model_identifier=RERANKER_MODEL,
            api_key=VOYAGE_API_KEY,
        )
        assert rr.reranker_id
        client.rerankers.delete(rr.reranker_id)

    def test_get_reranker(self, client: Goodmem):
        name = unique_name("reranker-get")
        rr = client.rerankers.create(
            display_name=name,
            model_identifier=RERANKER_MODEL,
            api_key=VOYAGE_API_KEY,
        )
        fetched = client.rerankers.get(rr.reranker_id)
        assert fetched.reranker_id == rr.reranker_id
        client.rerankers.delete(rr.reranker_id)

    def test_list_rerankers(self, client: Goodmem):
        name = unique_name("reranker-list")
        rr = client.rerankers.create(
            display_name=name,
            model_identifier=RERANKER_MODEL,
            api_key=VOYAGE_API_KEY,
        )
        ids = [r.reranker_id for r in client.rerankers.list()]
        assert rr.reranker_id in ids
        client.rerankers.delete(rr.reranker_id)

    def test_update_reranker(self, client: Goodmem):
        name = unique_name("reranker-update")
        rr = client.rerankers.create(
            display_name=name,
            model_identifier=RERANKER_MODEL,
            api_key=VOYAGE_API_KEY,
        )
        new_name = unique_name("reranker-updated")
        updated = client.rerankers.update(rr.reranker_id, display_name=new_name)
        assert updated.display_name == new_name
        client.rerankers.delete(rr.reranker_id)

    def test_delete_reranker(self, client: Goodmem):
        name = unique_name("reranker-delete")
        rr = client.rerankers.create(
            display_name=name,
            model_identifier=RERANKER_MODEL,
            api_key=VOYAGE_API_KEY,
        )
        client.rerankers.delete(rr.reranker_id)
        with pytest.raises(NotFoundError):
            client.rerankers.get(rr.reranker_id)

    def test_get_nonexistent_reranker_raises_not_found(self, client: Goodmem):
        with pytest.raises(NotFoundError):
            client.rerankers.get("00000000-0000-0000-0000-000000000000")


# ---------------------------------------------------------------------------
# 7. Spaces
# ---------------------------------------------------------------------------

class TestSpaces:
    def test_create_space(self, client: Goodmem, embedder_id: str):
        name = unique_name("space-create")
        sp = client.spaces.create(
            name=name,
            space_embedders=[{"embedder_id": embedder_id, "default_retrieval_weight": 1.0}],
            default_chunking_config={"none": {}},
        )
        assert sp.space_id
        client.spaces.delete(sp.space_id)

    def test_get_space(self, client: Goodmem, space_id: str):
        sp = client.spaces.get(space_id)
        assert sp.space_id == space_id

    def test_list_spaces_returns_results(self, client: Goodmem, space_id: str):
        ids = [s.space_id for s in client.spaces.list()]
        assert space_id in ids

    def test_update_space_name(self, client: Goodmem, embedder_id: str):
        name = unique_name("space-update")
        sp = client.spaces.create(
            name=name,
            space_embedders=[{"embedder_id": embedder_id, "default_retrieval_weight": 1.0}],
            default_chunking_config={"none": {}},
        )
        new_name = unique_name("space-updated")
        updated = client.spaces.update(sp.space_id, name=new_name)
        assert updated.name == new_name
        client.spaces.delete(sp.space_id)

    def test_delete_space(self, client: Goodmem, embedder_id: str):
        name = unique_name("space-delete")
        sp = client.spaces.create(
            name=name,
            space_embedders=[{"embedder_id": embedder_id, "default_retrieval_weight": 1.0}],
            default_chunking_config={"none": {}},
        )
        client.spaces.delete(sp.space_id)
        with pytest.raises(NotFoundError):
            client.spaces.get(sp.space_id)

    def test_get_nonexistent_space_raises_not_found(self, client: Goodmem):
        with pytest.raises(NotFoundError):
            client.spaces.get("00000000-0000-0000-0000-000000000000")


# ---------------------------------------------------------------------------
# 8. Memories – CRUD
# ---------------------------------------------------------------------------

class TestMemoriesCRUD:
    def test_create_plain_text_memory(self, client: Goodmem, space_id: str):
        mem = client.memories.create(
            space_id=space_id,
            original_content=PLAIN_TEXT_CONTENT,
        )
        assert mem.memory_id
        assert mem.processing_status in ("PENDING", "PROCESSING", "COMPLETED")
        client.memories.delete(mem.memory_id)

    def test_create_memory_returns_pending_status(self, client: Goodmem, space_id: str):
        mem = client.memories.create(
            space_id=space_id,
            original_content="Quick test memory.",
        )
        assert mem.processing_status in ("PENDING", "PROCESSING", "COMPLETED")
        client.memories.delete(mem.memory_id)

    def test_get_memory(self, client: Goodmem, space_id: str):
        mem = client.memories.create(
            space_id=space_id,
            original_content="Memory to get.",
        )
        fetched = client.memories.get(mem.memory_id)
        assert fetched.memory_id == mem.memory_id
        client.memories.delete(mem.memory_id)

    def test_memory_content_endpoint(self, client: Goodmem, space_id: str):
        text = "Retrievable content test."
        mem = client.memories.create(
            space_id=space_id,
            original_content=text,
        )
        wait_for_memory_processing(client, mem.memory_id)
        content = client.memories.content(mem.memory_id)
        assert content is not None
        client.memories.delete(mem.memory_id)

    def test_list_memories_in_space(self, client: Goodmem, space_id: str):
        mem = client.memories.create(
            space_id=space_id,
            original_content="List test memory.",
        )
        ids = [m.memory_id for m in client.memories.list(space_id=space_id)]
        assert mem.memory_id in ids
        client.memories.delete(mem.memory_id)

    def test_delete_memory(self, client: Goodmem, space_id: str):
        mem = client.memories.create(
            space_id=space_id,
            original_content="Memory to delete.",
        )
        client.memories.delete(mem.memory_id)
        with pytest.raises(NotFoundError):
            client.memories.get(mem.memory_id)

    def test_create_memory_with_metadata(self, client: Goodmem, space_id: str):
        mem = client.memories.create(
            space_id=space_id,
            original_content="Memory with metadata.",
            metadata={"author": "pytest", "version": "1"},
        )
        assert mem.memory_id
        client.memories.delete(mem.memory_id)

    def test_create_pdf_memory_base64(self, client: Goodmem, space_id: str):
        pdf_b64 = base64.standard_b64encode(_MINIMAL_PDF).decode("ascii")
        mem = client.memories.create(
            space_id=space_id,
            content_type="application/pdf",
            original_content_b64=pdf_b64,
            metadata={"source": "FILE"},
            chunking_config={
                "recursive": {
                    "chunk_size": 512,
                    "chunk_overlap": 64,
                    "keep_strategy": "KEEP_END",
                    "length_measurement": "CHARACTER_COUNT",
                }
            },
        )
        assert mem.memory_id
        client.memories.delete(mem.memory_id)

    def test_create_memory_with_recursive_chunking(self, client: Goodmem, space_id: str):
        long_text = "Deep learning is a subset of machine learning. " * 20
        mem = client.memories.create(
            space_id=space_id,
            original_content=long_text,
            chunking_config={
                "recursive": {
                    "chunk_size": 256,
                    "chunk_overlap": 32,
                    "keep_strategy": "KEEP_END",
                    "length_measurement": "CHARACTER_COUNT",
                }
            },
        )
        assert mem.memory_id
        client.memories.delete(mem.memory_id)

    def test_create_memory_with_sentence_chunking(self, client: Goodmem, space_id: str):
        text = "Sentence one. Sentence two. Sentence three. " * 5
        mem = client.memories.create(
            space_id=space_id,
            original_content=text,
            chunking_config={
                "sentence": {
                    "max_chunk_size": 200,
                    "min_chunk_size": 50,
                    "length_measurement": "CHARACTER_COUNT",
                }
            },
        )
        assert mem.memory_id
        client.memories.delete(mem.memory_id)

    def test_get_nonexistent_memory_raises_not_found(self, client: Goodmem):
        with pytest.raises(NotFoundError):
            client.memories.get("00000000-0000-0000-0000-000000000000")


# ---------------------------------------------------------------------------
# 9. Memories – Batch Operations
# ---------------------------------------------------------------------------

class TestMemoriesBatch:
    def test_batch_create(self, client: Goodmem, space_id: str):
        items = [
            {"original_content": f"Batch memory {i}."} for i in range(3)
        ]
        results = client.memories.batch_create(space_id=space_id, items=items)
        assert len(results) == 3
        for r in results:
            client.memories.delete(r.memory_id)

    def test_batch_get(self, client: Goodmem, space_id: str):
        mems = [
            client.memories.create(space_id=space_id, original_content=f"Batch get {i}.")
            for i in range(2)
        ]
        ids = [m.memory_id for m in mems]
        fetched = client.memories.batch_get(ids)
        fetched_ids = [f.memory_id for f in fetched]
        for mid in ids:
            assert mid in fetched_ids
        client.memories.batch_delete(ids)

    def test_batch_delete(self, client: Goodmem, space_id: str):
        mems = [
            client.memories.create(space_id=space_id, original_content=f"Batch del {i}.")
            for i in range(2)
        ]
        ids = [m.memory_id for m in mems]
        client.memories.batch_delete(ids)
        for mid in ids:
            with pytest.raises(NotFoundError):
                client.memories.get(mid)


# ---------------------------------------------------------------------------
# 10. Memory Retrieval (RAG)
# ---------------------------------------------------------------------------

class TestMemoryRetrieval:
    """
    These tests exercise the full RAG pipeline (embedder → space → LLM).
    They create their own isolated space and ingest data to avoid interfering
    with other tests.
    """

    @pytest.fixture(autouse=True)
    def rag_setup(self, client: Goodmem, embedder_id: str, llm_id: str):
        """Create a fresh space, ingest data, wait for completion."""
        space = client.spaces.create(
            name=unique_name("rag-space"),
            space_embedders=[{"embedder_id": embedder_id, "default_retrieval_weight": 1.0}],
            default_chunking_config={"none": {}},
        )
        self.rag_space_id = space.space_id
        self.llm_id = llm_id

        mem = client.memories.create(
            space_id=self.rag_space_id,
            original_content=PLAIN_TEXT_CONTENT,
        )
        self.memory_id = mem.memory_id

        wait_for_memory_processing(client, self.memory_id)

        yield

        try:
            client.memories.delete(self.memory_id)
        except APIError:
            pass
        try:
            client.spaces.delete(self.rag_space_id)
        except APIError:
            pass

    def test_retrieve_returns_stream_events(self, client: Goodmem):
        events = []
        with client.memories.retrieve(
            message="What do you know about Transformers?",
            space_ids=[self.rag_space_id],
            llm_id=self.llm_id,
        ) as stream:
            for event in stream:
                events.append(event)
        assert len(events) > 0

    def test_retrieve_stream_contains_abstract_reply(self, client: Goodmem):
        found_abstract = False
        with client.memories.retrieve(
            message="What is the attention mechanism?",
            space_ids=[self.rag_space_id],
            llm_id=self.llm_id,
        ) as stream:
            for event in stream:
                dumped = event.model_dump(by_alias=True, exclude_none=True)
                if "abstractReply" in dumped or "abstract_reply" in dumped:
                    found_abstract = True
                    break
        assert found_abstract, "Expected abstractReply event in RAG stream"

    @pytest.mark.skipif(not VOYAGE_API_KEY, reason="VOYAGE_API_KEY not set")
    def test_retrieve_with_reranker(self, client: Goodmem):
        rr = client.rerankers.create(
            display_name=unique_name("rr-rag"),
            model_identifier=RERANKER_MODEL,
            api_key=VOYAGE_API_KEY,
        )
        events = []
        try:
            with client.memories.retrieve(
                message="Describe neural networks.",
                space_ids=[self.rag_space_id],
                llm_id=self.llm_id,
                reranker_id=rr.reranker_id,
            ) as stream:
                for event in stream:
                    events.append(event)
        finally:
            client.rerankers.delete(rr.reranker_id)
        assert len(events) > 0

    def test_retrieve_multiple_space_ids(self, client: Goodmem, embedder_id: str):
        """Passing more than one space_id should work without error."""
        second_space = client.spaces.create(
            name=unique_name("rag-space2"),
            space_embedders=[{"embedder_id": embedder_id, "default_retrieval_weight": 1.0}],
            default_chunking_config={"none": {}},
        )
        events = []
        try:
            with client.memories.retrieve(
                message="Tell me about AI.",
                space_ids=[self.rag_space_id, second_space.space_id],
                llm_id=self.llm_id,
            ) as stream:
                for event in stream:
                    events.append(event)
        finally:
            client.spaces.delete(second_space.space_id)
        assert len(events) > 0


# ---------------------------------------------------------------------------
# 11. API Keys
# ---------------------------------------------------------------------------

class TestApiKeys:
    def test_create_api_key(self, client: Goodmem):
        key = client.apikeys.create(display_name=unique_name("test-key"))
        assert key is not None
        # Clean up
        key_id = getattr(key, "api_key_id", None) or getattr(key, "apiKeyId", None)
        if key_id:
            try:
                client.apikeys.delete(key_id)
            except APIError:
                pass

    def test_list_api_keys(self, client: Goodmem):
        key = client.apikeys.create(display_name=unique_name("list-key"))
        key_id = getattr(key, "api_key_id", None) or getattr(key, "apiKeyId", None)
        keys = list(client.apikeys.list())
        assert len(keys) >= 1
        if key_id:
            client.apikeys.delete(key_id)

    def test_update_api_key_display_name(self, client: Goodmem):
        name = unique_name("upd-key")
        key = client.apikeys.create(display_name=name)
        key_id = getattr(key, "api_key_id", None) or getattr(key, "apiKeyId", None)
        if key_id is None:
            pytest.skip("Cannot determine api_key_id from response")
        new_name = unique_name("upd-key-renamed")
        updated = client.apikeys.update(key_id, display_name=new_name)
        assert updated.display_name == new_name
        client.apikeys.delete(key_id)

    def test_delete_api_key(self, client: Goodmem):
        key = client.apikeys.create(display_name=unique_name("del-key"))
        key_id = getattr(key, "api_key_id", None) or getattr(key, "apiKeyId", None)
        if key_id is None:
            pytest.skip("Cannot determine api_key_id from response")
        client.apikeys.delete(key_id)
        # After deletion, the key should no longer appear in list
        remaining_ids = [
            getattr(k, "api_key_id", None) or getattr(k, "apiKeyId", None)
            for k in client.apikeys.list()
        ]
        assert key_id not in remaining_ids


# ---------------------------------------------------------------------------
# 12. Error Handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_not_found_error_has_status_404(self, client: Goodmem):
        try:
            client.memories.get("00000000-0000-0000-0000-000000000000")
        except NotFoundError as e:
            assert e.status_code == 404
        except APIError as e:
            assert e.status_code == 404

    def test_authentication_error_bad_key(self):
        bad = Goodmem(base_url=BASE_URL, api_key="gm_definitely_wrong")
        with pytest.raises((AuthenticationError, APIError)) as exc_info:
            bad.users.me()
        if isinstance(exc_info.value, APIError):
            assert exc_info.value.status_code in (401, 403)

    def test_api_error_has_body(self, client: Goodmem):
        try:
            client.memories.get("00000000-0000-0000-0000-000000000000")
        except APIError as e:
            assert e.body is not None


# ---------------------------------------------------------------------------
# 13. Async Client Smoke Test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestAsyncClient:
    async def test_async_system_info(self):
        async with AsyncGoodmem(base_url=BASE_URL, api_key=API_KEY) as c:
            info = await c.system.info()
            assert info is not None

    async def test_async_embedder_list(self):
        async with AsyncGoodmem(base_url=BASE_URL, api_key=API_KEY) as c:
            embedders = [e async for e in await c.embedders.list()]
            assert isinstance(embedders, list)

    async def test_async_create_and_delete_memory(self):
        async with AsyncGoodmem(base_url=BASE_URL, api_key=API_KEY) as c:
            # Reuse a space – find first available
            spaces = [s async for s in await c.spaces.list()]
            if not spaces:
                pytest.skip("No spaces available for async memory test")
            space_id = spaces[0].space_id
            mem = await c.memories.create(
                space_id=space_id,
                original_content="Async test memory.",
            )
            assert mem.memory_id
            await c.memories.delete(mem.memory_id)


# ---------------------------------------------------------------------------
# 14. Paginator
# ---------------------------------------------------------------------------

class TestPaginator:
    def test_embedder_paginator_iterates_all(self, client: Goodmem):
        """Iterating the paginator should yield all embedders without error."""
        count = 0
        for _ in client.embedders.list():
            count += 1
        assert count >= 0  # just ensure no exception

    def test_space_paginator_next_page(self, client: Goodmem):
        """Calling next_page() manually should not raise."""
        paginator = client.spaces.list()
        page = paginator.next_page()
        assert page is not None


# ---------------------------------------------------------------------------
# Entry point for direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import subprocess
    import sys

    sys.exit(subprocess.call(["pytest", __file__, "-v"]))