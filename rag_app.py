"""
GoodMem RAG Application
========================
A fully self-contained RAG app using the GoodMem Python SDK.

Features
--------
- Interactive CLI chat loop
- Plain-text knowledge base (add passages inline or from a .txt file)
- OpenAI embedder (text-embedding-3-large) + OpenAI LLM (gpt-4o-mini)
- Streaming responses from the RAG agent
- Full lifecycle management (setup → ingest → query → teardown)

Prerequisites
-------------
    pip install goodmem openai

Environment variables
---------------------
    GOODMEM_BASE_URL   – GoodMem server URL, e.g. http://localhost:8080
    GOODMEM_API_KEY    – GoodMem API key  (prefix gm_)
    OPENAI_API_KEY     – OpenAI API key

Quick start
-----------
    python rag_app.py                        # interactive mode
    python rag_app.py --load my_docs.txt     # load a text file then chat
    python rag_app.py --query "What is X?"   # single-shot query
    python rag_app.py --teardown             # delete all GoodMem resources
"""

import argparse
import json
import os
import sys
import time

from goodmem import Goodmem
from goodmem import APIError, NotFoundError

# ---------------------------------------------------------------------------
# Configuration  (override via environment variables)
# ---------------------------------------------------------------------------

BASE_URL       = os.environ.get("GOODMEM_BASE_URL", "http://localhost:8080")
API_KEY        = os.environ.get("GOODMEM_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

EMBEDDER_MODEL   = "text-embedding-3-large"
LLM_MODEL        = "gpt-4o-mini"
SPACE_NAME       = "rag-app-space"
EMBEDDER_NAME    = "rag-app-embedder"
LLM_NAME         = "rag-app-llm"

# File to persist resource IDs between runs
STATE_FILE = ".rag_app_state.json"

# ---------------------------------------------------------------------------
# Built-in sample knowledge base  (used when no external file is provided)
# ---------------------------------------------------------------------------

SAMPLE_KNOWLEDGE = [
    (
        "Artificial Intelligence (AI) is the simulation of human intelligence by machines. "
        "It encompasses machine learning, natural language processing, computer vision, "
        "and robotics. AI systems can learn from data, reason about problems, and adapt "
        "to new situations without explicit programming."
    ),
    (
        "Machine learning is a subset of AI that enables systems to learn and improve "
        "from experience. Key paradigms include supervised learning (learning from labeled "
        "data), unsupervised learning (finding hidden patterns), and reinforcement learning "
        "(learning through rewards and penalties)."
    ),
    (
        "Transformers are a neural network architecture introduced in the paper "
        "'Attention is All You Need' (2017). They rely on a self-attention mechanism "
        "to capture long-range dependencies in sequences. Transformers power modern LLMs "
        "such as GPT-4, Claude, and Gemini."
    ),
    (
        "Retrieval-Augmented Generation (RAG) combines information retrieval with text "
        "generation. A query is first used to retrieve relevant passages from a knowledge "
        "base; those passages are then injected into the LLM prompt so the model can "
        "answer questions grounded in external, up-to-date information."
    ),
    (
        "Vector databases store embeddings — numerical representations of text — and "
        "support fast similarity search. When a user submits a query, it is embedded "
        "into the same vector space and the nearest stored vectors are retrieved. "
        "Popular vector stores include Pinecone, Weaviate, Qdrant, and pgvector."
    ),
    (
        "Python is a high-level, interpreted programming language known for its readability "
        "and simplicity. It is the dominant language in data science and AI, supported by "
        "libraries such as NumPy, Pandas, PyTorch, and TensorFlow."
    ),
    (
        "GoodMem is a memory layer for AI agents. It manages embedders, LLMs, and "
        "memory spaces as first-class resources. Developers ingest plain text or files, "
        "and GoodMem handles chunking, embedding, storage, and retrieval — returning "
        "an LLM-generated answer grounded in the stored memories."
    ),
]


# ---------------------------------------------------------------------------
# Colour helpers (gracefully degrade on Windows)
# ---------------------------------------------------------------------------

def _colour(code: str, text: str) -> str:
    if sys.stdout.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text

def green(t):  return _colour("92", t)
def cyan(t):   return _colour("96", t)
def yellow(t): return _colour("93", t)
def red(t):    return _colour("91", t)
def bold(t):   return _colour("1",  t)
def dim(t):    return _colour("2",  t)


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# Setup: create embedder, LLM, space
# ---------------------------------------------------------------------------

def setup(client: Goodmem) -> dict:
    """
    Create (or reuse) an embedder, LLM, and space.
    Returns a state dict with resource IDs.
    """
    state = load_state()

    # ── Embedder ──────────────────────────────────────────────────────────
    embedder_id = state.get("embedder_id")
    if embedder_id:
        try:
            client.embedders.get(embedder_id)
            print(dim(f"  Reusing embedder  {embedder_id}"))
        except NotFoundError:
            embedder_id = None

    if not embedder_id:
        print(f"  Creating embedder  ({EMBEDDER_MODEL}) …")
        emb = client.embedders.create(
            display_name=EMBEDDER_NAME,
            model_identifier=EMBEDDER_MODEL,
            api_key=OPENAI_API_KEY,
        )
        embedder_id = emb.embedder_id
        print(green(f"  ✓ Embedder created  {embedder_id}"))

    # ── LLM ───────────────────────────────────────────────────────────────
    llm_id = state.get("llm_id")
    if llm_id:
        try:
            client.llms.get(llm_id)
            print(dim(f"  Reusing LLM        {llm_id}"))
        except NotFoundError:
            llm_id = None

    if not llm_id:
        print(f"  Creating LLM       ({LLM_MODEL}) …")
        llm = client.llms.create(
            display_name=LLM_NAME,
            model_identifier=LLM_MODEL,
            api_key=OPENAI_API_KEY,
        )
        llm_id = llm.llm_id
        print(green(f"  ✓ LLM created       {llm_id}"))

    # ── Space ─────────────────────────────────────────────────────────────
    space_id = state.get("space_id")
    if space_id:
        try:
            client.spaces.get(space_id)
            print(dim(f"  Reusing space      {space_id}"))
        except NotFoundError:
            space_id = None

    if not space_id:
        print(f"  Creating space …")
        sp = client.spaces.create(
            name=SPACE_NAME,
            space_embedders=[{"embedder_id": embedder_id, "default_retrieval_weight": 1.0}],
            default_chunking_config={"none": {}},
        )
        space_id = sp.space_id
        print(green(f"  ✓ Space created     {space_id}"))

    state = {
        "embedder_id": embedder_id,
        "llm_id":      llm_id,
        "space_id":    space_id,
        "memory_ids":  state.get("memory_ids", []),
    }
    save_state(state)
    return state


# ---------------------------------------------------------------------------
# Ingest knowledge
# ---------------------------------------------------------------------------

def _wait_for_memory(client: Goodmem, memory_id: str, timeout: int = 90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        mem = client.memories.get(memory_id)
        status = mem.processing_status
        if status == "COMPLETED":
            return
        if status in ("FAILED", "ERROR"):
            print(red(f"  ✗ Memory {memory_id} processing failed: {status}"))
            return
        time.sleep(2)
    print(yellow(f"  ⚠ Timed out waiting for memory {memory_id}"))


def ingest_texts(client: Goodmem, state: dict, texts: list[str]) -> dict:
    """Ingest a list of text passages into the space."""
    space_id = state["space_id"]
    memory_ids = list(state.get("memory_ids", []))

    print(f"\n{bold('Ingesting')} {len(texts)} passage(s) into space …")
    for i, text in enumerate(texts, 1):
        preview = text[:60].replace("\n", " ")
        print(f"  [{i}/{len(texts)}] {dim(preview + '…')}")
        mem = client.memories.create(
            space_id=space_id,
            original_content=text,
        )
        memory_ids.append(mem.memory_id)

    # Wait for all memories to be processed
    print("  Waiting for embeddings to be ready …", end="", flush=True)
    for mid in memory_ids[-len(texts):]:
        _wait_for_memory(client, mid)
        print(".", end="", flush=True)
    print(f" {green('done')}")

    state["memory_ids"] = memory_ids
    save_state(state)
    return state


def ingest_file(client: Goodmem, state: dict, filepath: str) -> dict:
    """
    Ingest passages from a plain-text file.
    Passages are split on blank lines (double newline).
    """
    print(f"\nLoading file: {bold(filepath)}")
    with open(filepath, encoding="utf-8") as f:
        raw = f.read()

    # Split on blank lines; filter empty chunks
    passages = [p.strip() for p in raw.split("\n\n") if p.strip()]
    print(f"  Found {len(passages)} passage(s)")
    return ingest_texts(client, state, passages)


# ---------------------------------------------------------------------------
# Query / RAG
# ---------------------------------------------------------------------------

def query(client: Goodmem, state: dict, question: str) -> str:
    """
    Run the RAG pipeline for a single question.
    Returns the full abstract reply text.
    """
    space_id = state["space_id"]
    llm_id   = state["llm_id"]

    full_reply = ""
    raw_events = []

    with client.memories.retrieve(
        message=question,
        space_ids=[space_id],
        llm_id=llm_id,
    ) as stream:
        for event in stream:
            raw_events.append(event)
            dumped = event.model_dump(by_alias=True, exclude_none=True)

            # Stream token-by-token if delta events are present
            if "delta" in dumped:
                token = dumped["delta"].get("text", "")
                print(token, end="", flush=True)
                full_reply += token

            # Capture the final abstract reply
            elif "abstractReply" in dumped:
                text = dumped["abstractReply"].get("text", "")
                if not full_reply:
                    # No streaming deltas — print the full reply now
                    print(text, end="", flush=True)
                full_reply = text

    print()  # newline after streamed output
    return full_reply


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------

def teardown(client: Goodmem, state: dict):
    """Delete all GoodMem resources created by this app."""
    print(f"\n{bold('Tearing down')} GoodMem resources …")

    for mid in state.get("memory_ids", []):
        try:
            client.memories.delete(mid)
            print(dim(f"  Deleted memory   {mid}"))
        except APIError:
            pass

    if sid := state.get("space_id"):
        try:
            client.spaces.delete(sid)
            print(dim(f"  Deleted space    {sid}"))
        except APIError:
            pass

    if lid := state.get("llm_id"):
        try:
            client.llms.delete(lid)
            print(dim(f"  Deleted LLM      {lid}"))
        except APIError:
            pass

    if eid := state.get("embedder_id"):
        try:
            client.embedders.delete(eid)
            print(dim(f"  Deleted embedder {eid}"))
        except APIError:
            pass

    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)

    print(green("  ✓ Teardown complete"))


# ---------------------------------------------------------------------------
# Interactive chat loop
# ---------------------------------------------------------------------------

COMMANDS = {
    "/help":     "Show this help",
    "/add":      "Add a text passage  →  /add <your text>",
    "/load":     "Load passages from a file  →  /load path/to/file.txt",
    "/list":     "List ingested memory IDs",
    "/clear":    "Delete all memories and re-ingest sample knowledge",
    "/teardown": "Delete all GoodMem resources and exit",
    "/quit":     "Exit the app (resources are preserved)",
}


def print_help():
    print(f"\n{bold('Commands:')}")
    for cmd, desc in COMMANDS.items():
        print(f"  {cyan(cmd):20s}  {desc}")
    print()


def chat_loop(client: Goodmem, state: dict):
    print(f"\n{bold('=' * 58)}")
    print(f"  {bold('GoodMem RAG Chat')}  —  type {cyan('/help')} for commands")
    print(f"  Knowledge base: {len(state.get('memory_ids', []))} passage(s) loaded")
    print(f"{bold('=' * 58)}\n")

    while True:
        try:
            user_input = input(f"{bold(cyan('You'))} › ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{dim('Exiting …')}")
            break

        if not user_input:
            continue

        # ── Commands ──────────────────────────────────────────────────────
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd   = parts[0].lower()
            arg   = parts[1] if len(parts) > 1 else ""

            if cmd == "/help":
                print_help()

            elif cmd == "/add":
                if not arg:
                    print(yellow("  Usage: /add <text passage>"))
                else:
                    state = ingest_texts(client, state, [arg])

            elif cmd == "/load":
                if not arg:
                    print(yellow("  Usage: /load <filepath>"))
                elif not os.path.exists(arg):
                    print(red(f"  File not found: {arg}"))
                else:
                    state = ingest_file(client, state, arg)

            elif cmd == "/list":
                ids = state.get("memory_ids", [])
                if ids:
                    print(f"  {len(ids)} memory IDs:")
                    for mid in ids:
                        print(f"    {dim(mid)}")
                else:
                    print(yellow("  No memories ingested yet."))

            elif cmd == "/clear":
                print("  Deleting all memories …")
                for mid in state.get("memory_ids", []):
                    try:
                        client.memories.delete(mid)
                    except APIError:
                        pass
                state["memory_ids"] = []
                save_state(state)
                print(green("  ✓ Cleared. Re-ingesting sample knowledge …"))
                state = ingest_texts(client, state, SAMPLE_KNOWLEDGE)

            elif cmd in ("/teardown", "/exit-and-delete"):
                teardown(client, state)
                sys.exit(0)

            elif cmd in ("/quit", "/exit", "/q"):
                print(dim("Resources preserved. Goodbye!"))
                break

            else:
                print(yellow(f"  Unknown command: {cmd}  (try /help)"))

        # ── RAG query ─────────────────────────────────────────────────────
        else:
            if not state.get("memory_ids"):
                print(yellow(
                    "  No knowledge loaded yet.\n"
                    "  Use /add <text> or /load <file> to add knowledge first."
                ))
                continue

            print(f"\n{bold(green('Assistant'))} › ", end="", flush=True)
            try:
                query(client, state, user_input)
            except APIError as e:
                print(red(f"\n  API error {e.status_code}: {e.body}"))
            print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GoodMem RAG application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--load", metavar="FILE",
        help="Load passages from a plain-text file (split on blank lines) then enter chat",
    )
    parser.add_argument(
        "--query", metavar="QUESTION",
        help="Run a single-shot query and exit",
    )
    parser.add_argument(
        "--no-sample", action="store_true",
        help="Skip loading the built-in sample knowledge base on first run",
    )
    parser.add_argument(
        "--teardown", action="store_true",
        help="Delete all GoodMem resources created by this app and exit",
    )
    args = parser.parse_args()

    # ── Validate environment ───────────────────────────────────────────────
    missing = [v for v in ("GOODMEM_BASE_URL", "GOODMEM_API_KEY", "OPENAI_API_KEY")
               if not os.environ.get(v)]
    if missing:
        print(red("Missing required environment variables:"))
        for v in missing:
            print(f"  export {v}=...")
        sys.exit(1)

    client = Goodmem(base_url=BASE_URL, api_key=API_KEY)

    # ── Teardown mode ─────────────────────────────────────────────────────
    if args.teardown:
        state = load_state()
        if not state:
            print(yellow("No state file found — nothing to tear down."))
        else:
            teardown(client, state)
        return

    # ── Setup ─────────────────────────────────────────────────────────────
    print(f"\n{bold('GoodMem RAG App')} — setting up resources …")
    state = setup(client)

    # First run: load sample knowledge unless --no-sample
    if not state.get("memory_ids") and not args.no_sample:
        print(f"\n{bold('First run')} — loading built-in sample knowledge …")
        state = ingest_texts(client, state, SAMPLE_KNOWLEDGE)

    # ── Load file if provided ─────────────────────────────────────────────
    if args.load:
        if not os.path.exists(args.load):
            print(red(f"File not found: {args.load}"))
            sys.exit(1)
        state = ingest_file(client, state, args.load)

    # ── Single-shot query mode ────────────────────────────────────────────
    if args.query:
        print(f"\n{bold('Query:')} {args.query}")
        print(f"\n{bold(green('Answer'))} › ", end="", flush=True)
        query(client, state, args.query)
        print()
        return

    # ── Interactive chat ──────────────────────────────────────────────────
    chat_loop(client, state)


if __name__ == "__main__":
    main()