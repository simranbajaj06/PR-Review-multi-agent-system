# PR Reviewer — Multi-Agent System

An AI-powered GitHub PR review agent, built incrementally toward the architecture
described in [*Designing an AI Pull-Request Review Agent*](https://www.antern.co/blogs/production-grade-ai-pr-review-agent/).
That write-up derives a full production design — parallel specialist reviewers,
retrieval-grounded context, a confidence-weighted human-in-the-loop gate, and a
single event spine for audit/cost/tracing. This repo builds toward that design
one working phase at a time rather than all at once.

**Current status: multi-agent review with retrieval (RAG).** Four specialist
agents (security, quality, tests, docs) review each diff in parallel, an
aggregator merges their structured findings, and each specialist is grounded in
relevant existing code retrieved from a pgvector index of the repo. The
human-in-the-loop gate is computed but not yet enforced, and the queue is still
in-memory — see [Roadmap](#roadmap) and [Known limitations](#known-limitations).

## What it does today

**On a pull request** (`opened`, `synchronize`, `reopened`, `ready_for_review`):

1. Verifies the HMAC-SHA256 signature and drops duplicate deliveries
   (`X-GitHub-Delivery`)
2. Enqueues the event and returns `200` immediately
3. A background worker fetches the PR diff from the GitHub API
4. **Retrieval:** relevant existing code (callers, helpers, neighbouring
   functions) is pulled from the RAG index and rendered as a "reference code" block
5. A LangGraph graph fans out to four specialists in parallel (security,
   quality, tests, docs). Each sees the diff plus the reference code and returns
   structured `Finding`s
6. The aggregator dedupes findings, computes an overall confidence, applies the
   confidence/severity gate, and renders one markdown review
7. The review is posted back to the PR as a comment

**On a push to the default branch:**

1. The push event is turned into an `IndexJob` and enqueued (pushes to other
   branches, branch deletions and malformed payloads are ignored)
2. The worker runs `IndexUpdater.sync(repo, sha)`, which diffs the last indexed
   commit against the new one, re-chunks only the changed Python files, and
   re-embeds only chunks whose content hash changed

```mermaid
flowchart LR
    GH[GitHub webhook] --> Router[FastAPI webhook router]
    Router --> Handler[WebhookHandler<br/>verify + dedupe + dispatch]
    Handler -->|pull_request| Queue[In-memory queue]
    Handler -->|push to default branch| Queue
    Queue --> Worker[worker thread]

    Worker -->|PullRequestEvent| Fetch[fetch_diff]
    Fetch --> Reviewer[GroqReviewer]
    Reviewer --> Retriever[ContextRetriever]
    Retriever <--> PG[(Postgres + pgvector)]
    Reviewer --> Graph[LangGraph:<br/>4 specialists in parallel]
    Graph --> Agg[Aggregator<br/>dedupe + confidence gate]
    Agg --> Post[post_comment]
    Post --> GH

    Worker -->|IndexJob| Updater[IndexUpdater.sync]
    Updater --> Chunk[Chunk changed files]
    Chunk --> Embed[Qwen3 embedder]
    Embed --> PG
```

## How retrieval works

The index tracks the repo's **default branch**, so it represents the code a PR
is being merged into.

- **Chunking** (`rag/code_chunker.py`): Python source is split with `ast` into
  function, method and class-header chunks (`Class.method` naming). Nested
  functions stay in their parent; oversized chunks are split into `#partN`;
  duplicate names get a `#L<line>` suffix. Only `.py` files are indexed.
- **Embedding** (`adapters/embeddings/qwen3_embedder.py`): `Qwen3-Embedding-0.6B`
  runs locally via sentence-transformers (free, no API). Vectors are 1024-dim and
  normalised. Chunks are embedded with a `# path :: name (kind)` header so
  queries can match on location as well as body.
- **Storage** (`adapters/storage/pg_chunk_store.py`): Postgres + pgvector.
  `code_chunks` holds one row per chunk, unique on `(repo, file_path, chunkname)`,
  with `kind`, line range, `content`, `content_hash`, `commit_sha` and the
  `embedding vector(1024)`. `index_state` stores the commit the index currently
  reflects, per repo.
- **Incremental indexing** (`rag/indexer.py`, `rag/updater.py`): first run is a
  full tarball snapshot; later runs use GitHub's compare API to find changed
  files and only re-chunk those. Per chunk: new or changed hash → embed and
  upsert; unchanged content → skip embedding but refresh line numbers and
  `commit_sha`; chunk gone → delete. It falls back to a full index after a
  force-push or when more than 60 files changed. An empty chunk list never
  wipes the index.
- **Retrieval** (`rag/retriever.py`): the diff is parsed into hunks; each
  meaningful hunk (file path, enclosing `def`/`class`, changed lines) becomes a
  query. Hits are filtered by similarity, deduped, and kept best-first within a
  character budget. Chunks that overlap a changed hunk are dropped, because
  they're the pre-change version of code the diff already shows; untouched
  chunks in the same file are kept.
- **Prompting** (`agents/base_agent.py`): reference code is added after the diff
  under a clearly labelled section, and the system prompt tells the model it is
  background only — findings must be about lines in the diff.

## Architecture

The code follows a ports-and-adapters (hexagonal) layout, which keeps each new
capability additive:

- **`domain/`** — framework-agnostic core. `interfaces.py` defines the ports
  (`WebhookVerifier`, `ReviewQueue`, `VCSClient`, `Reviewer`); `models.py`
  defines the data shapes (`PullRequestEvent`, `IndexJob`, `Review`);
  `contracts.py` defines the `Finding` / `SpecialistOutput` contract every
  specialist returns; `services/` holds logic that depends only on those ports.
- **`rag/`** — the retrieval layer. `ports.py` defines `Embedder`, `ChunkStore`
  and `RepoSource`; the chunker, indexer, updater and retriever depend only on
  those ports.
- **`agents/`** — one small module per specialist (a prompt + an `AgentType`),
  all built on the shared `make_specialist_node` in `base_agent.py`.
- **`orchestrator/`** — the LangGraph topology (`graph.py`), shared state
  (`state.py`) and the merge/gate logic (`aggregator.py`).
- **`adapters/`** — concrete implementations of each port (GitHub, queue, LLM,
  embeddings, Postgres). Swappable without touching `domain/` or `rag/`.
- **`api/`** — the FastAPI HTTP layer; delegates everything to `WebhookHandler`.
- **`workers/`** — drains the queue and dispatches review and index jobs.
- **`main.py`** — the composition root; the only file that imports both ports
  and concrete adapters and wires them together. RAG is built once at startup
  and is optional.

## Project structure

```
.
├── main.py                              # composition root
├── scripts/
│   └── index_repo.py                    # CLI: index / refresh a repo, optional test query
└── src/pr_reviewer/
    ├── config.py                        # settings (webhook secret, DATABASE_URL)
    ├── api/
    │   └── webhook_router.py            # POST /webhook/github
    ├── domain/
    │   ├── models.py                    # PullRequestEvent, IndexJob, Review
    │   ├── interfaces.py                # WebhookVerifier, ReviewQueue, VCSClient, Reviewer
    │   ├── contracts.py                 # Finding, Severity, AgentType, SpecialistOutput
    │   └── services/
    │       └── webhook_handler.py       # verify -> dedupe -> dispatch -> parse -> enqueue
    ├── rag/
    │   ├── ports.py                     # Embedder, ChunkStore, RepoSource
    │   ├── code_chunker.py              # Python source -> function/method/class chunks
    │   ├── indexer.py                   # chunks -> embeddings -> DB (hash-based skipping)
    │   ├── updater.py                   # incremental sync of the index to a commit
    │   └── retriever.py                 # diff -> queries -> relevant chunks -> prompt block
    ├── agents/
    │   ├── base_agent.py                # shared LLM call, JSON-mode parsing, retries
    │   ├── security_agent.py
    │   ├── quality_agent.py
    │   ├── tests_agent.py
    │   └── docs_agent.py
    ├── orchestrator/
    │   ├── graph.py                     # parallel fan-out to specialists -> aggregate
    │   ├── state.py                     # ReviewState (incl. retrieved context)
    │   └── aggregator.py                # dedupe, overall confidence, gate, markdown
    ├── adapters/
    │   ├── github/
    │   │   ├── signature_verifier.py    # HMAC-SHA256 verification
    │   │   ├── payload_parser.py        # pull_request JSON -> PullRequestEvent
    │   │   ├── push_parser.py           # push JSON -> IndexJob
    │   │   ├── github_client.py         # fetch_diff, post_comment
    │   │   └── github_repo_fetcher.py   # RepoSource: snapshot, compare, read_file
    │   ├── queue/
    │   │   └── in_memory_queue.py       # process-local queue (reviews + index jobs)
    │   ├── llm/
    │   │   └── grok_reviewer.py         # Reviewer: retrieval + LangGraph + ChatGroq
    │   ├── embeddings/
    │   │   └── qwen3_embedder.py        # local Qwen3-Embedding-0.6B
    │   └── storage/
    │       └── pg_chunk_store.py        # ChunkStore on Postgres + pgvector
    └── workers/
        └── review_worker.py             # dispatches PullRequestEvent / IndexJob
```

## Setup

```bash
git clone <this repo>
cd <this repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

RAG additionally needs these packages (add them to `requirements.txt` if they
aren't there): `psycopg[binary]>=3.1`, `pgvector`, `numpy`,
`sentence-transformers>=2.7`, `transformers>=4.51`, `requests`. The first run
downloads the embedding model (about 1.2 GB).

Create a `.env` file in the project root:

```bash
GITHUB_TOKEN=ghp_...              # classic PAT; repo scope (or public_repo for public repos)
GITHUB_WEBHOOK_SECRET=...          # must match the secret set on the GitHub webhook
GROQ_API_KEY=gsk_...               # console.groq.com/keys
DATABASE_URL=postgresql://user:pass@host:5432/dbname   # optional: enables RAG
```

`DATABASE_URL` is optional. Without it the app runs diff-only (no retrieval, no
push indexing) and `/health` reports `rag_enabled: false`.

### Database

RAG needs Postgres with the `pgvector` extension and two tables:

- `code_chunks` — `repo`, `file_path`, `chunkname`, `kind`, `start_line`,
  `end_line`, `content`, `content_hash`, `commit_sha`, `embedding vector(1024)`,
  `updated_at`, with a unique constraint on `(repo, file_path, chunkname)`
- `index_state` — `repo` (unique), `indexed_sha`, `updated_at`

The `vector(1024)` size must match the embedding model; the app checks this at
startup and refuses to start on a mismatch.

### Index the repo once

Before relying on retrieval, build the initial index so the first review isn't
diff-only:

```bash
python -m scripts.index_repo owner/name
python -m scripts.index_repo owner/name --ref main
python -m scripts.index_repo owner/name --query "where is auth checked?"   # test search
```

It's cheap to re-run: later runs only process files changed since the last
indexed commit. After this, pushes to the default branch keep the index current
automatically.

### Run

```bash
uvicorn main:app --reload
```

On startup you should see `RAG enabled (embedding dim=1024)`.

Expose it for GitHub to reach (local dev):

```bash
ngrok http 8000
```

Then, on the repo: **Settings → Webhooks → Add webhook**
- Payload URL: `https://<your-ngrok-domain>/webhook/github`
- Content type: `application/json`
- Secret: same value as `GITHUB_WEBHOOK_SECRET`
- Events: **Pull requests** and **Pushes**

Open or update a PR and watch the worker log a retrieval summary, the review,
and a posted comment. Merge to the default branch and watch it log an index job.

## Tuning knobs

| Setting | Where | Default |
|---|---|---|
| Max chunk size before splitting | `code_chunker.MAX_CHUNK_CHARS` | 6,000 chars |
| Max file size indexed | `code_chunker.MAX_FILE_BYTES` | 500 KB |
| Full re-index above this many changed files | `updater.MAX_INCREMENTAL_FILES` | 60 |
| Results per query | `ContextRetriever(k_per_query=…)` | 4 |
| Minimum similarity | `ContextRetriever(min_similarity=…)` | 0.3 |
| Max queries per PR | `ContextRetriever(max_queries=…)` | 12 |
| Context budget | `ContextRetriever(max_context_chars=…)` | 12,000 chars |
| Keep chunks overlapping changed hunks | `ContextRetriever(include_modified_chunks=…)` | `False` |
| Human-review confidence threshold | `aggregator.CONFIDENCE_THRESHOLD` | 0.6 |
| Retries per specialist call | `base_agent.MAX_RETRIES` | 2 |

The similarity cutoff and budget are starting guesses; tune them against real
PRs using the `context: N chunks [...]` log line printed for each review.

## Known limitations

- **Retrieval failures aren't handled yet.** If the embedder or database errors
  during retrieval, that PR's review fails (logged by the worker) instead of
  falling back to a diff-only review.
- **Unindexed repos get no context.** A PR on a repo that hasn't been indexed
  reviews diff-only, silently.
- **`MAX_DIFF_CHARS` is defined but not applied**, so very large diffs aren't
  truncated before being sent to the model.
- **The human-review gate isn't enforced.** The aggregator flags reviews with a
  critical finding or low confidence, but the worker still posts every review.
- **One worker thread.** A long index job (especially the first full index)
  blocks PR reviews queued behind it. Run `scripts.index_repo` first.
- **Python only.** The chunker indexes `.py` files; other languages are ignored
  for retrieval.
- **The index follows the default branch.** Line ranges used to drop
  pre-change chunks assume the index matches the PR's base commit, which is
  approximate if the base moved since the last push.
- **In-memory queue and dedupe set** reset on restart and don't scale past one
  process.

## Roadmap

Tracking the phases from the design doc this project follows. Unchecked items
are the next honest steps, not aspirational scope creep — each one is meant to
land as a single additive change against the existing ports.

- [x] Webhook ingress: HMAC verification, idempotency, fast ack
- [x] End-to-end loop: fetch diff → LLM review → post comment
- [x] Orchestration: LangGraph graph with parallel fan-out and a join
- [x] Structured `Finding` contract (severity, category, file/line, confidence,
      rationale)
- [x] Multi-agent split: security / quality / tests / docs specialists running
      in parallel, merged by an aggregator
- [x] Retrieval (RAG): chunking, local embeddings, pgvector storage, incremental
      indexing on push, diff-driven retrieval, grounded specialist prompts
- [ ] RAG hardening: fall back to diff-only on retrieval errors, handle
      unindexed repos, apply the diff size cap, tune similarity/budget on real PRs
- [ ] Confidence-weighted human-in-the-loop gate: the decision is computed
      today; still to build is holding flagged reviews instead of auto-posting,
      and routing them to a review queue
- [ ] Queue: swap `InMemoryReviewQueue` for Redis-backed, decouple the worker
      into its own process (and separate index jobs from review jobs)
- [ ] Observability: an event log for every LLM/tool call, feeding a trace
      view, an audit trail, and cost tracking
- [ ] Reliability layer: retries with backoff, circuit breakers, and
      fault-tolerant handling around the Groq and GitHub calls