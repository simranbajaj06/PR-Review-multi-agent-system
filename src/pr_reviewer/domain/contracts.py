"""Shared contract that every specialist agent returns and the aggregator
consumes. Keeping this as one small, boring module is what lets you add a
5th specialist later without touching the aggregator's merge logic - it
only ever deals with Finding objects, never agent-specific shapes.
"""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AgentType(str, Enum):
    SECURITY = "security"
    QUALITY = "quality"
    TESTS = "tests"
    DOCS = "docs"


class Finding(BaseModel):
    """One reviewable issue, attributed to the specialist that raised it."""
    agent_type: AgentType
    severity: Severity
    category: str = Field(description="short label, e.g. 'injection', 'missing-test'")
    summary: str = Field(description="one-line description of the issue")
    file: Optional[str] = Field(default=None, description="path relative to repo root")
    line: Optional[int] = Field(default=None, description="line number in the new file, if known")
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(description="why this is a real issue - cite the code")
    suggestion: Optional[str] = Field(default=None, description="concrete fix, if obvious")


class SpecialistOutput(BaseModel):
    """The structured-output shape each specialist LLM call must return.
    Empty findings list is a valid, expected response - it means 'this
    concern is clean', not 'the agent failed'.
    """
    findings: list[Finding] = Field(default_factory=list)
