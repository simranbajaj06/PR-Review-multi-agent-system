"""Inbound HTTP layer. Only knows about HTTP - all real logic is delegated
to WebhookHandler, which this route doesn't even know the internals of.
"""
from fastapi import APIRouter, Header, HTTPException, Request

from src.pr_reviewer.domain.services.webhook_handler import WebhookHandler

router = APIRouter()


@router.post("/webhook/github")
async def github_webhook(
    request: Request,
    x_github_event: str | None = Header(default=None),
    x_hub_signature_256: str | None = Header(default=None),
    x_github_delivery: str | None = Header(default=None),
):
    body = await request.body()
    payload = await request.json()
    handler: WebhookHandler = request.app.state.webhook_handler

    # Event-type filtering now lives in the handler (after signature
    # verification), so adding another event type doesn't touch this file.
    try:
        return handler.handle(
            body,
            x_hub_signature_256,
            payload,
            delivery_id=x_github_delivery,
            event_type=x_github_event or "",
        )
    except WebhookHandler.InvalidSignature:
        raise HTTPException(status_code=401, detail="invalid signature")