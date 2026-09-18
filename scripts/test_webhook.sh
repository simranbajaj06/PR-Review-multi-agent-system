#!/usr/bin/env bash
# Simulates GitHub sending a signed pull_request webhook to your local server.
# Usage: GITHUB_WEBHOOK_SECRET=your_secret ./scripts/test_webhook.sh
set -euo pipefail

SECRET="${GITHUB_WEBHOOK_SECRET:?Set GITHUB_WEBHOOK_SECRET first}"
URL="${WEBHOOK_URL:-http://127.0.0.1:8000/webhook/github}"
BODY=$(cat "$(dirname "$0")/sample_payload.json")
SIG="sha256=$(printf '%s' "$BODY" | openssl dgst -sha256 -hmac "$SECRET" | sed 's/^.* //')"

curl -s -X POST "$URL" \
  -H "Content-Type: application/json" \
  -H "X-GitHub-Event: pull_request" \
  -H "X-Hub-Signature-256: $SIG" \
  -d "$BODY"
echo
