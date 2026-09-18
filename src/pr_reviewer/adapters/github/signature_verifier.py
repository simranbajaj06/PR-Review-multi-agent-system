"""Verifies GitHub's HMAC-SHA256 webhook signature (the X-Hub-Signature-256 header)."""
import hashlib
import hmac

from pr_reviewer.domain.interfaces import WebhookVerifier


class GitHubSignatureVerifier(WebhookVerifier):
    def __init__(self, secret: str):
        self._secret = secret.encode()

    def verify(self, payload_body: bytes, signature_header: str | None) -> bool:
        if not signature_header or not signature_header.startswith("sha256="):
            return False
        expected = "sha256=" + hmac.new(
            self._secret, payload_body, hashlib.sha256
        ).hexdigest()
        # constant-time comparison to avoid timing attacks
        return hmac.compare_digest(expected, signature_header)
