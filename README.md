# PR Reviewer Agent — Step 1: PR notification

This is just the first slice of the project: get notified the instant a PR is
opened or updated on **one repo**, verified and queued for review. The LLM
review step comes later.

## 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

## 2. Configure your secret

```bash
cp .env.example .env
```

Edit `.env` and set `GITHUB_WEBHOOK_SECRET` to a long random string, e.g.:

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

Load it into your shell before running the app:

```bash
export $(cat .env | xargs)
```

## 3. Run the server

```bash
uvicorn pr_reviewer.main:app --reload --port 8000
```

Check it's up: `curl http://127.0.0.1:8000/health` → `{"status": "ok"}`

## 4. Expose it to the internet (local dev only)

GitHub can't reach `localhost`, so tunnel it. Either works:

```bash
# Option A: smee.io (no account needed for a quick test)
npx smee-client --url https://smee.io/<your-channel> --target http://127.0.0.1:8000/webhook/github

# Option B: ngrok
ngrok http 8000
```

Note the public URL it gives you (e.g. `https://abcd1234.ngrok-free.app/webhook/github`).

## 5. Register the webhook on your repo

In your repo → **Settings → Webhooks → Add webhook**:

- **Payload URL**: the public URL from step 4
- **Content type**: `application/json`
- **Secret**: the exact same value as `GITHUB_WEBHOOK_SECRET`
- **Which events**: choose "Let me select individual events" → check only **Pull requests**

Save it. GitHub sends a `ping` event immediately — you should see a `200`
next to the webhook in GitHub's UI under "Recent Deliveries".

## 6. Test it

Open a real PR on the repo, or simulate one locally without touching GitHub:

```bash
chmod +x scripts/test_webhook.sh
GITHUB_WEBHOOK_SECRET=<same secret> ./scripts/test_webhook.sh
```

You should see `{"status": "accepted", "pr_number": 42}` and, in the server
logs, a line like:

```
[queue] enqueued PR #42 (octocat/example-repo): 'Add retry logic to payment client'
```

## What happens next

`app.state.review_queue` now holds reviewable PR events. The next step is a
worker that pulls events off this queue, fetches the diff via the GitHub API,
and sends it to an LLM for review — that's a new adapter
(`adapters/llm/...`) and a new domain service, without touching anything
built here.

## Project layout

```
src/pr_reviewer/
├── main.py                        # composition root — wires everything together
├── config.py                      # env var loading
├── api/webhook_router.py          # HTTP layer only
├── domain/
│   ├── models.py                  # PullRequestEvent
│   ├── interfaces.py              # WebhookVerifier, ReviewQueue (the ports)
│   └── services/webhook_handler.py  # verify -> parse -> filter -> enqueue
└── adapters/
    ├── github/                    # concrete GitHub implementations
    └── queue/                     # concrete queue implementation (in-memory for now)
```
