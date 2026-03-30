"""
Agent Orchestrator

Revision metadata: ``LAST_UPDATED`` and ``model_name`` are defined at module scope.

Routes requests between specialized agents based on intent classification.
Manages the multi-agent workflow and aggregates responses.

SECURITY NOTES (for Unifai demo):
- Inter-agent calls are not authenticated
- No privilege verification between agent calls
- Token passed but never validated

POLICY VIOLATIONS NOTICED:
- [Authenticate Agent Interactions] Missing inter-agent authentication is a policy violation.
  You MUST implement authentication for every agent-to-agent call. Currently, the
  _agent_token is generated but never validated by receiving agents, and is_internal=True
  bypasses privilege checks. This must be remediated immediately.
- [Enforce Approved LLM] The model "deepseek-r1" may not be on the approved LLM allow list.
  Please replace it with an approved LLM from your organization's allow list.
"""

import logging
import os
import re
import base64
import json
import uuid
import datetime
from typing import Any, Optional

from .tech_support import TechSupportAgent
from .finance import FinanceAgent
from .file_processor import FileProcessorAgent
from .auth.agent_auth import AgentAuthenticator, AgentIdentity
from llm.openrouter import OpenRouterClient

logger = logging.getLogger(__name__)

model_name = "deepseek-r1"
LAST_UPDATED = "2025-06-08"

# ---------------------------------------------------------------------------
# PII redaction patterns (zero-tolerance categories)
# ---------------------------------------------------------------------------
_PII_PATTERNS = [
    # SSN
    (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), '[REDACTED SSN]'),
    # Year of Birth (standalone 4-digit year 1900-2099)
    (re.compile(r'\b(19|20)\d{2}\b'), '[REDACTED YOB]'),
    # Personal Phone Number
    (re.compile(r'\b(\+?\d[\d\s\-().]{7,}\d)\b'), '[REDACTED PHONE]'),
    # Email
    (re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'), '[REDACTED EMAIL]'),
    # Home Address (basic heuristic: number + street)
    (re.compile(r'\b\d+\s+[A-Za-z0-9\s,\.]{5,50}(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl)\b', re.IGNORECASE), '[REDACTED ADDRESS]'),
    # Passport Number
    (re.compile(r'\b[A-Z]{1,2}\d{6,9}\b'), '[REDACTED PASSPORT]'),
    # Drivers License (generic)
    (re.compile(r'\b[A-Z]{1,2}\d{5,8}\b'), '[REDACTED DL]'),
    # Taxpayer Identification Number / EIN
    (re.compile(r'\b\d{2}-\d{7}\b'), '[REDACTED TIN]'),
    # Credit Card Number
    (re.compile(r'\b(?:\d[ \-]?){13,16}\b'), '[REDACTED CC]'),
    # Financial Account Number (8-17 digits)
    (re.compile(r'\b\d{8,17}\b'), '[REDACTED ACCOUNT]'),
    # IP Address
    (re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'), '[REDACTED IP]'),
    # MAC Address
    (re.compile(r'\b([0-9A-Fa-f]{2}[:\-]){5}[0-9A-Fa-f]{2}\b'), '[REDACTED MAC]'),
    # Employee ID / School ID (generic badge patterns)
    (re.compile(r'\b(EMP|SCH|STU|EID|SID)[#\-]?\d{4,10}\b', re.IGNORECASE), '[REDACTED ID]'),
    # Vehicle Identification Number
    (re.compile(r'\b[A-HJ-NPR-Z0-9]{17}\b'), '[REDACTED VIN]'),
    # GPS / Fine Location coordinates
    (re.compile(r'\b-?\d{1,3}\.\d{4,},\s*-?\d{1,3}\.\d{4,}\b'), '[REDACTED LOCATION]'),
    # Ethnicity keywords
    (re.compile(r'\b(African American|Asian|Caucasian|Hispanic|Latino|Native American|Pacific Islander|Middle Eastern)\b', re.IGNORECASE), '[REDACTED ETHNICITY]'),
    # Sexual Orientation keywords
    (re.compile(r'\b(heterosexual|homosexual|bisexual|gay|lesbian|queer|pansexual|asexual)\b', re.IGNORECASE), '[REDACTED SEXUAL ORIENTATION]'),
]

# ---------------------------------------------------------------------------
# Singapore PII redaction patterns
# ---------------------------------------------------------------------------
_SG_PII_PATTERNS = [
    # NRIC / FIN Number
    (re.compile(r'\b[STFGM]\d{7}[A-Z]\b'), 'REDACTED'),
    # Passport Number
    (re.compile(r'\b[A-Z]{1,2}\d{6,9}\b'), 'REDACTED'),
    # Work Permit / Student Pass / Government ID
    (re.compile(r'\b(WP|SP|EP|DP|LTVP|PEP|EntrePass)[\s\-]?\d{6,12}\b', re.IGNORECASE), 'REDACTED'),
    # Date of Birth
    (re.compile(r'\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b'), 'REDACTED'),
    # Personal Mobile / Home Phone
    (re.compile(r'\b(\+65[\s\-]?)?\d{4}[\s\-]?\d{4}\b'), 'REDACTED'),
    # Email
    (re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'), 'REDACTED'),
    # Bank Account / Credit Card / Debit Card
    (re.compile(r'\b\d{8,19}\b'), 'REDACTED'),
    # CPF Account Number
    (re.compile(r'\bCPF[\s\-]?\d{7,9}\b', re.IGNORECASE), 'REDACTED'),
    # IP Address
    (re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'), 'REDACTED'),
    # MAC Address
    (re.compile(r'\b([0-9A-Fa-f]{2}[:\-]){5}[0-9A-Fa-f]{2}\b'), 'REDACTED'),
    # GPS Coordinates
    (re.compile(r'\b-?\d{1,3}\.\d{4,},\s*-?\d{1,3}\.\d{4,}\b'), 'REDACTED'),
    # SingPass / MyInfo / Digital Identity identifiers (heuristic)
    (re.compile(r'\b(singpass|myinfo|corppass)[\s\-]?id[\s\-]?:?\s*\S+\b', re.IGNORECASE), 'REDACTED'),
    # Session / Auth tokens (Bearer / JWT-like)
    (re.compile(r'\b(Bearer\s+)?[A-Za-z0-9\-_]{20,}\.[A-Za-z0-9\-_]{20,}\.[A-Za-z0-9\-_]{20,}\b'), 'REDACTED'),
    # Ethnicity / Race / Religion / Sexual Orientation
    (re.compile(r'\b(Chinese|Malay|Indian|Eurasian|Caucasian|heterosexual|homosexual|bisexual|gay|lesbian|queer|Buddhist|Christian|Muslim|Hindu|Sikh|Taoist|Catholic)\b', re.IGNORECASE), 'REDACTED'),
    # Salary / Financial amounts
    (re.compile(r'\bS?\$\s?\d[\d,]*(\.\d{2})?\b'), 'REDACTED'),
    # Full Name heuristic (Title + Name)
    (re.compile(r'\b(Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)+\b'), 'REDACTED'),
    # Residential / Mailing Address (Singapore postal code)
    (re.compile(r'\b(Blk|Block|No\.?)?\s*\d+\s+[A-Za-z\s]+,?\s*(#\d{2}-\d{2,4})?,?\s*Singapore\s*\d{6}\b', re.IGNORECASE), 'REDACTED'),
    # Postal code standalone
    (re.compile(r'\bSingapore\s+\d{6}\b', re.IGNORECASE), 'REDACTED'),
]

# ---------------------------------------------------------------------------
# Suspicious content patterns for uploaded files
# ---------------------------------------------------------------------------
_SUSPICIOUS_COMMANDS = re.compile(
    r'\b(alias|ripgrep|rg|curl|rm|echo|dd|git|tar|chmod|chown|fsck|wget|nc|netcat|ncat|'
    r'bash|sh|zsh|fish|ksh|csh|tcsh|python|python3|perl|ruby|php|node|nodejs|'
    r'exec|eval|system|popen|subprocess|os\.system|os\.popen|'
    r'powershell|cmd\.exe|wscript|cscript|mshta|'
    r'sudo|su|passwd|useradd|userdel|usermod|groupadd|'
    r'iptables|ufw|firewall|nmap|masscan|'
    r'base64|xxd|hexdump|od|strings|'
    r'mount|umount|fdisk|mkfs|fsck|'
    r'crontab|at|batch|'
    r'kill|killall|pkill|'
    r'export|unset|env|printenv|'
    r'find|locate|which|whereis|'
    r'ping|traceroute|dig|nslookup|host|'
    r'ssh|scp|sftp|ftp|telnet|'
    r'openssl|gpg|'
    r'docker|kubectl|helm|'
    r'apt|apt-get|yum|dnf|pip|npm|gem|cargo)\b',
    re.IGNORECASE
)

_BASE64_PATTERN = re.compile(r'\b(?:[A-Za-z0-9+/]{4}){4,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?\b')

_LEETSPEAK_PATTERN = re.compile(
    r'\b[a-z0-9]*(?:(?:3(?=.*[a-z])|4(?=.*[a-z])|1(?=.*[a-z])|0(?=.*[a-z])|5(?=.*[a-z])|7(?=.*[a-z]))[a-z0-9]*){2,}\b',
    re.IGNORECASE
)

_BINARY_PATTERN = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]{4,}')

_SHELL_SHEBANG = re.compile(r'^#!\s*/\S+', re.MULTILINE)

_DYNAMIC_CODE_PATTERNS = re.compile(
    r'(eval\s*\(|exec\s*\(|subprocess\s*\(\s*.*shell\s*=\s*True|'
    r'os\.system\s*\(|os\.popen\s*\(|__import__\s*\(|'
    r'compile\s*\(.*exec|'
    r'<script[^>]*>.*?</script>)',
    re.IGNORECASE | re.DOTALL
)

# ---------------------------------------------------------------------------
# Hidden / invisible prompt patterns
# ---------------------------------------------------------------------------
_INVISIBLE_PROMPT_PATTERNS = [
    # Zero-width characters
    re.compile(r'[\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad]'),
    # Very small font HTML tags
    re.compile(r'<\s*(?:font|span)[^>]*(?:size\s*=\s*["\']?[01]["\']?|font-size\s*:\s*[01]px)[^>]*>', re.IGNORECASE),
    # White-on-white or invisible color CSS
    re.compile(r'color\s*:\s*(?:white|#fff(?:fff)?|rgba?\(255,255,255)', re.IGNORECASE),
    # Base64 encoded content
    _BASE64_PATTERN,
    # Leetspeak
    _LEETSPEAK_PATTERN,
]


def _redact_pii(text: str) -> str:
    """Redact zero-tolerance PII categories from text."""
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _redact_sg_pii(text: str) -> str:
    """Redact Singapore PII categories from text."""
    for pattern, replacement in _SG_PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _remove_suspicious_content(text: str) -> str:
    """Remove suspicious commands, binaries, and encoded content from file text."""
    # Remove binary-like sequences
    text = _BINARY_PATTERN.sub('<suspicious_content_removed>', text)
    # Remove shell shebangs
    text = _SHELL_SHEBANG.sub('<suspicious_content_removed>', text)
    # Attempt to decode and check base64 blobs
    def _check_base64(m):
        try:
            decoded = base64.b64decode(m.group(0)).decode('utf-8', errors='ignore')
            if _SUSPICIOUS_COMMANDS.search(decoded):
                return '<suspicious_content_removed>'
        except Exception:
            pass
        return '<suspicious_content_removed>'
    text = _BASE64_PATTERN.sub(_check_base64, text)
    # Remove suspicious shell/system commands
    text = _SUSPICIOUS_COMMANDS.sub('<suspicious_content_removed>', text)
    return text


def _sanitize_llm_input(text: str) -> str:
    """
    Sanitize text before sending to LLM:
    - Remove hidden/invisible prompts
    - Remove base64 encoded suspicious content
    - Remove leetspeak patterns
    - Remove shell commands and binaries
    - Redact PII
    """
    # Remove zero-width and invisible characters
    text = re.sub(r'[\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad]', '', text)
    # Remove HTML invisible font tricks
    text = re.sub(r'<\s*(?:font|span)[^>]*(?:size\s*=\s*["\']?[01]["\']?|font-size\s*:\s*[01]px)[^>]*>.*?</\s*(?:font|span)\s*>', '', text, flags=re.IGNORECASE | re.DOTALL)
    # Remove white-on-white style blocks
    text = re.sub(r'<[^>]*color\s*:\s*(?:white|#fff(?:fff)?|rgba?\(255,255,255)[^>]*>.*?</[^>]+>', '', text, flags=re.IGNORECASE | re.DOTALL)
    # Decode and check base64
    def _check_b64(m):
        try:
            decoded = base64.b64decode(m.group(0)).decode('utf-8', errors='ignore')
            if _SUSPICIOUS_COMMANDS.search(decoded) or _DYNAMIC_CODE_PATTERNS.search(decoded):
                return '<suspicious_content_removed>'
        except Exception:
            pass
        return '<suspicious_content_removed>'
    text = _BASE64_PATTERN.sub(_check_b64, text)
    # Remove shell commands
    text = _SUSPICIOUS_COMMANDS.sub('<suspicious_content_removed>', text)
    # Redact PII before sending to LLM
    text = _redact_pii(text)
    return text.strip()


def _sanitize_llm_response(text: str) -> str:
    """
    Sanitize LLM response:
    - Remove lines containing eval, exec, subprocess(shell=True), JS eval, bash eval
    - Remove dynamic code execution primitives
    """
    lines = text.splitlines()
    safe_lines = []
    for line in lines:
        if _DYNAMIC_CODE_PATTERNS.search(line):
            logger.warning("Removed dangerous code execution pattern from LLM response.")
            continue
        safe_lines.append(line)
    return '\n'.join(safe_lines)


def _log_llm_interaction(direction: str, content: Any, extra: dict = None):
    """Log every LLM interaction (input and output)."""
    log_entry = {
        "llm_interaction": direction,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "interaction_id": str(uuid.uuid4()),
        "content_length": len(str(content)) if content else 0,
    }
    if extra:
        log_entry.update(extra)
    logger.info("LLM interaction logged", extra=log_entry)


def _mask_pii_for_log(text: str) -> str:
    """Mask PII in log strings."""
    return _redact_pii(str(text))


class AgentOrchestrator:
    """
    Central orchestrator that routes requests to appropriate agents.

    The orchestrator:
    1. Classifies user intent
    2. Routes to the appropriate agent
    3. Handles inter-agent communication
    4. Aggregates and returns responses
    """

    def __init__(self):
        self.llm_client = OpenRouterClient()
        self.authenticator = AgentAuthenticator()

        # Initialize agents
        self.tech_support = TechSupportAgent(self.llm_client)
        self.finance = FinanceAgent(self.llm_client)
        self.file_processor = FileProcessorAgent()

        # Agent registry with privilege levels
        self.agents = {
            "tech_support": {
                "agent": self.tech_support,
                "privilege": "low",
                "description": "General technical support and queries"
            },
            "finance": {
                "agent": self.finance,
                "privilege": "high",
                "description": "Financial data and reports"
            },
            "file_processor": {
                "agent": self.file_processor,
                "privilege": "medium",
                "description": "File processing and analysis"
            }
        }

        # Token for inter-agent communication — loaded from environment variable
        self._agent_token = os.environ.get("AGENT_INTERNAL_TOKEN", "")
        if not self._agent_token:
            logger.warning("AGENT_INTERNAL_TOKEN environment variable is not set. Inter-agent authentication will fail.")

    async def process(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Process incoming request and route to appropriate agent(s).

        Args:
            context: Request context including message, files, and metadata

        Returns:
            Response dictionary with agent output
        """
        user_message = context.get("user_message", "")
        file_contents = context.get("file_contents", [])

        # Sanitize user message before any processing
        user_message = _sanitize_llm_input(user_message)
        context["user_message"] = user_message

        logger.info(
            "Orchestrator processing request",
            extra={
                "message_length": len(user_message),
                "file_count": len(file_contents),
                # PII masked in log preview
                "context_preview": _mask_pii_for_log(str(context))[:200]
            }
        )

        # Determine which agent should handle the request
        intent = await self._classify_intent(user_message, file_contents)

        # Route to appropriate agent
        if intent == "finance":
            return await self._route_to_finance(context)
        elif intent == "file_analysis":
            return await self._route_to_file_processor(context)
        else:
            return await self._route_to_tech_support(context)

    async def _classify_intent(
        self,
        message: str,
        file_contents: list
    ) -> str:
        """
        Classify the user's intent to determine routing.

        Returns one of: 'finance', 'file_analysis', 'tech_support'
        """
        # Simple keyword-based classification for demo
        message_lower = message.lower()

        finance_keywords = [
            "finance", "financial", "budget", "revenue", "expense",
            "profit", "loss", "quarterly", "annual report", "earnings",
            "balance sheet", "income statement", "cash flow"
        ]

        if any(keyword in message_lower for keyword in finance_keywords):
            return "finance"

        if file_contents:
            return "file_analysis"

        return "tech_support"

    async def _route_to_tech_support(
        self,
        context: dict[str, Any]
    ) -> dict[str, Any]:
        """Route request to tech support agent."""
        # Create internal caller identity
        caller = AgentIdentity(
            agent_id="orchestrator",
            agent_name="Orchestrator",
            privilege_level="system",
            is_internal=True  # Flag that bypasses auth
        )

        # VULNERABILITY: Token passed but never validated by receiving agent
        headers = {"X-Agent-Token": self._agent_token}

        response = await self.tech_support.handle(
            context=context,
            caller=caller,
            headers=headers
        )

        return response

    async def _route_to_finance(
        self,
        context: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Route request to finance agent.

        VULNERABILITY: This method allows routing to high-privilege agent
        without proper authentication or authorization checks.
        """
        # Create internal caller identity
        # VULNERABILITY: is_internal=True bypasses privilege checks
        caller = AgentIdentity(
            agent_id="orchestrator",
            agent_name="Orchestrator",
            privilege_level="system",
            is_internal=True
        )

        # Token passed but receiver doesn't validate
        headers = {"X-Agent-Token": self._agent_token}

        logger.info(
            "Routing to finance agent",
            extra={
                "caller": caller.agent_id,
                "privilege": caller.privilege_level,
                # Token preview masked — do not log secrets
                "token_preview": "[REDACTED]"
            }
        )

        response = await self.finance.handle(
            context=context,
            caller=caller,
            headers=headers
        )

        return response

    async def _route_to_file_processor(
        self,
        context: dict[str, Any]
    ) -> dict[str, Any]:
        """Route request to file processor agent."""
        file_contents = context.get("file_contents", [])

        if not file_contents:
            return {
                "response": "No files were provided to analyze.",
                "agent": "file_processor"
            }

        # Process files and get analysis
        analyses = []
        for file_data in file_contents:
            extracted = file_data.get("extracted_content", "")

            # Remove suspicious commands and binaries from file content
            extracted = _remove_suspicious_content(extracted)

            # Redact Singapore PII from file content
            extracted = _redact_sg_pii(extracted)

            # Redact zero-tolerance PII from file content
            extracted = _redact_pii(extracted)

            analyses.append(f"File: {file_data.get('filename')}\n{extracted}")

        combined_content = "\n\n".join(analyses)

        # Get the user's actual question — sanitize before sending to LLM
        user_question = _sanitize_llm_input(context.get("user_message", ""))

        # Sanitize combined file content before sending to LLM
        combined_content_sanitized = _sanitize_llm_input(combined_content)

        system_prompt = (
            "You are a helpful document analyst. Answer the user's questions based on the "
            "provided document content. Be direct and specific - if they ask for specific "
            "information, provide it exactly as it appears in the document."
        )
        user_prompt = (
            f"Document Content:\n{combined_content_sanitized}\n\n"
            f"User Question: {user_question}\n\n"
            "Please answer the user's question based on the document content above."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Log LLM input interaction
        _log_llm_interaction("input", messages, extra={"agent": "file_processor"})

        # Get LLM analysis of file contents
        analysis = await self.llm_client.chat(messages=messages)

        # Sanitize LLM response
        analysis = _sanitize_llm_response(analysis)

        # Log LLM output interaction
        _log_llm_interaction("output", analysis, extra={"agent": "file_processor"})

        return {
            "response": analysis,
            "agent": "file_processor",
            "files_processed": len(file_contents)
        }

    async def escalate_from_tech_support(
        self,
        query: str,
        tech_support_context: dict
    ) -> dict[str, Any]:
        """
        Handle escalation from tech support to finance agent.

        This method is called when tech support needs to access
        financial data on behalf of a user.

        VULNERABILITY: No verification that tech support has permission
        to access finance agent on behalf of this user.
        """
        # Sanitize query before processing
        query = _sanitize_llm_input(query)

        # VULNERABILITY: Direct escalation without privilege verification
        escalation_context = {
            "user_message": query,
            "escalated_from": "tech_support",
            "original_context": tech_support_context,
            "escalation_reason": "Financial data requested"
        }

        logger.info(
            "Escalating from tech support to finance",
            extra={
                "query": _mask_pii_for_log(query),
                "original_context": _mask_pii_for_log(str(tech_support_context))[:100]
            }
        )

        return await self._route_to_finance(escalation_context)