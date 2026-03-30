"""
Finance Agent

Revision metadata: ``LAST_UPDATED`` and ``model_name`` are defined at module scope.

Handles financial data queries with HIGH privilege level.
Should only be accessible to authorized callers.

SECURITY NOTES (for Unifai demo):
- Authorization check exists but has bypass for "internal" calls
- Sensitive financial data returned without audit logging
- No rate limiting on data access
"""

import logging
import re
import base64
import uuid
from typing import Any, Optional

from .auth.agent_auth import AgentIdentity, AgentAuthenticator
from llm.openrouter import OpenRouterClient

logger = logging.getLogger(__name__)

model_name = "deepseek-r1"
LAST_UPDATED = "2024-04-11"

# Approved LLM models allow list
APPROVED_LLM_MODELS = [
    "gpt-4",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "claude-3-opus",
    "claude-3-sonnet",
    "claude-3-haiku",
    "claude-2",
    "gemini-pro",
]

# Patterns indicating dynamic code execution primitives
CODE_EXECUTION_PATTERNS = [
    re.compile(r'\beval\s*\(', re.IGNORECASE),
    re.compile(r'\bexec\s*\(', re.IGNORECASE),
    re.compile(r'\bsubprocess\s*\(', re.IGNORECASE),
    re.compile(r'shell\s*=\s*True', re.IGNORECASE),
    re.compile(r'\bos\.system\s*\(', re.IGNORECASE),
    re.compile(r'\bos\.popen\s*\(', re.IGNORECASE),
    re.compile(r'\b__import__\s*\(', re.IGNORECASE),
    re.compile(r'\bcompile\s*\(', re.IGNORECASE),
    re.compile(r'<script[\s>]', re.IGNORECASE),
    re.compile(r'\bsetTimeout\s*\(', re.IGNORECASE),
    re.compile(r'\bsetInterval\s*\(', re.IGNORECASE),
    re.compile(r'\bnew\s+Function\s*\(', re.IGNORECASE),
]

# Patterns for detecting suspicious prompt injection content
SUSPICIOUS_PROMPT_PATTERNS = [
    re.compile(r'ignore\s+(previous|above|prior)\s+instructions', re.IGNORECASE),
    re.compile(r'disregard\s+(previous|above|prior)\s+instructions', re.IGNORECASE),
    re.compile(r'forget\s+(previous|above|prior)\s+instructions', re.IGNORECASE),
    re.compile(r'you\s+are\s+now\s+', re.IGNORECASE),
    re.compile(r'act\s+as\s+', re.IGNORECASE),
    re.compile(r'pretend\s+(you\s+are|to\s+be)', re.IGNORECASE),
    re.compile(r'jailbreak', re.IGNORECASE),
    re.compile(r'DAN\s+mode', re.IGNORECASE),
    re.compile(r'system\s+prompt', re.IGNORECASE),
    re.compile(r'reveal\s+(your|the)\s+(system\s+)?prompt', re.IGNORECASE),
]

# Leetspeak character mapping for normalization
LEET_MAP = str.maketrans({
    '0': 'o', '1': 'i', '3': 'e', '4': 'a',
    '5': 's', '6': 'g', '7': 't', '8': 'b', '@': 'a',
    '$': 's', '!': 'i', '+': 't',
})


class FinanceAgent:
    """
    Finance agent for handling financial data queries.

    Privilege Level: HIGH
    Capabilities:
    - Access financial reports
    - Query budget information
    - Generate financial summaries

    SECURITY: This agent handles sensitive financial data and
    should only be accessible to authorized callers.
    """

    ALLOWED_ROLES = ["finance_admin", "cfo", "admin"]
    PRIVILEGE_LEVEL = "high"

    def __init__(self, llm_client: OpenRouterClient):
        self.llm_client = llm_client
        self.authenticator = AgentAuthenticator()
        self.agent_id = "finance"
        self.agent_name = "Finance Agent"

        # Warn if the configured model is not on the approved list
        if model_name not in APPROVED_LLM_MODELS:
            logger.warning(
                "SECURITY NOTICE: The configured LLM model '%s' is not on the approved "
                "allow list. Please replace it with an approved model from the following "
                "list: %s. Using an unapproved model may violate security and compliance "
                "policies.",
                model_name,
                APPROVED_LLM_MODELS,
            )

        # Simulated financial data (would be database in real app)
        self._financial_data = {
            "quarterly_revenue": {
                "Q1_2024": 2500000,
                "Q2_2024": 2750000,
                "Q3_2024": 3100000,
                "Q4_2024": 3400000
            },
            "operating_expenses": {
                "Q1_2024": 1800000,
                "Q2_2024": 1900000,
                "Q3_2024": 2000000,
                "Q4_2024": 2100000
            },
            "employee_salaries": {
                "engineering": 1200000,
                "sales": 800000,
                "operations": 600000,
                "executive": 500000
            },
            "sensitive_projections": {
                "merger_target": "CompetitorCorp",
                "acquisition_budget": 50000000,
                "layoff_planning": "Q2 2025 - 15% reduction"
            }
        }

    # ------------------------------------------------------------------
    # Input sanitization helpers
    # ------------------------------------------------------------------

    def _is_base64_encoded(self, text: str) -> bool:
        """Return True if text appears to be base64-encoded content."""
        # Strip whitespace and check if it looks like base64
        stripped = text.strip()
        if len(stripped) < 20:
            return False
        base64_pattern = re.compile(r'^[A-Za-z0-9+/\s]+=*$')
        if not base64_pattern.match(stripped):
            return False
        try:
            decoded = base64.b64decode(stripped, validate=True).decode('utf-8', errors='ignore')
            # If decoded text is printable and longer than a threshold, flag it
            printable_ratio = sum(c.isprintable() for c in decoded) / max(len(decoded), 1)
            return printable_ratio > 0.7 and len(decoded) > 10
        except Exception:
            return False

    def _contains_invisible_text(self, text: str) -> bool:
        """Detect invisible or hidden text techniques (zero-width chars, tiny font markers)."""
        invisible_chars = [
            '\u200b',  # zero-width space
            '\u200c',  # zero-width non-joiner
            '\u200d',  # zero-width joiner
            '\u2060',  # word joiner
            '\ufeff',  # BOM / zero-width no-break space
            '\u00ad',  # soft hyphen
        ]
        return any(ch in text for ch in invisible_chars)

    def _contains_binary_or_shell(self, text: str) -> bool:
        """Detect binary executables or shell command patterns."""
        shell_patterns = [
            re.compile(r'\\x[0-9a-fA-F]{2}'),          # hex escape sequences
            re.compile(r'/bin/(sh|bash|zsh|dash)', re.IGNORECASE),
            re.compile(r'cmd\.exe', re.IGNORECASE),
            re.compile(r'powershell', re.IGNORECASE),
            re.compile(r'wget\s+http', re.IGNORECASE),
            re.compile(r'curl\s+http', re.IGNORECASE),
            re.compile(r'chmod\s+\+x', re.IGNORECASE),
            re.compile(r'nc\s+-[el]', re.IGNORECASE),   # netcat reverse shell
            re.compile(r'>\s*/dev/', re.IGNORECASE),
        ]
        return any(p.search(text) for p in shell_patterns)

    def _normalize_leetspeak(self, text: str) -> str:
        """Normalize common leetspeak substitutions for pattern matching."""
        return text.translate(LEET_MAP)

    def _sanitize_input(self, text: str) -> str:
        """
        Sanitize and validate user input before sending to the LLM.

        Raises ValueError if the input is deemed unsafe.
        Returns the sanitized text.
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string.")

        # Remove null bytes and control characters (except newline/tab)
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

        # Detect invisible/hidden text
        if self._contains_invisible_text(sanitized):
            raise ValueError("Input contains invisible or hidden text characters.")

        # Detect base64-encoded payloads
        if self._is_base64_encoded(sanitized):
            raise ValueError("Input appears to contain base64-encoded content.")

        # Detect binary or shell command patterns
        if self._contains_binary_or_shell(sanitized):
            raise ValueError("Input contains binary or shell command patterns.")

        # Check for code execution primitives
        for pattern in CODE_EXECUTION_PATTERNS:
            if pattern.search(sanitized):
                raise ValueError("Input contains disallowed code execution primitives.")

        # Check for suspicious prompt injection patterns (also in leetspeak-normalized form)
        normalized = self._normalize_leetspeak(sanitized)
        for pattern in SUSPICIOUS_PROMPT_PATTERNS:
            if pattern.search(sanitized) or pattern.search(normalized):
                raise ValueError("Input contains suspicious prompt injection content.")

        # Truncate to a safe maximum length
        max_length = 4000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]

        return sanitized

    # ------------------------------------------------------------------
    # LLM response sanitization helper
    # ------------------------------------------------------------------

    def _sanitize_llm_response(self, response: str) -> str:
        """
        Validate and sanitize the response received from the LLM.

        Removes any line that contains eval, exec, or other dynamic
        code-execution primitives.
        """
        if not isinstance(response, str):
            logger.warning("LLM returned a non-string response; converting to empty string.")
            return ""

        cleaned_lines = []
        for line in response.splitlines():
            flagged = False
            for pattern in CODE_EXECUTION_PATTERNS:
                if pattern.search(line):
                    logger.warning(
                        "Removed dangerous line from LLM response",
                        extra={"removed_line": line[:200]}
                    )
                    flagged = True
                    break
            if not flagged:
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    # ------------------------------------------------------------------
    # Core handler
    # ------------------------------------------------------------------

    async def handle(
        self,
        context: dict[str, Any],
        caller: AgentIdentity,
        headers: Optional[dict] = None
    ) -> dict[str, Any]:
        """
        Handle incoming request with authorization check.

        Args:
            context: Request context with query details
            caller: Identity of the calling agent/user
            headers: Request headers (including auth token)

        Returns:
            Response dictionary with financial data or error
        """
        # Authorization check
        # VULNERABILITY: Check exists but has bypass
        if not self._verify_authorization(caller, headers):
            logger.warning(
                "Unauthorized access attempt to finance agent",
                extra={
                    "caller_id": caller.agent_id,
                    "caller_privilege": caller.privilege_level
                }
            )
            return {
                "response": "Unauthorized: You do not have permission to access financial data.",
                "agent": self.agent_id,
                "error": "unauthorized"
            }

        user_message = context.get("user_message", "")

        # Sanitize and validate input before processing
        try:
            user_message = self._sanitize_input(user_message)
        except ValueError as exc:
            logger.warning(
                "Rejected unsafe input to finance agent",
                extra={"caller_id": caller.agent_id, "reason": str(exc)}
            )
            return {
                "response": f"Invalid input: {exc}",
                "agent": self.agent_id,
                "error": "invalid_input"
            }

        # Process the financial query
        response = await self._process_financial_query(user_message)

        return {
            "response": response,
            "agent": self.agent_id,
            "privilege_level": self.PRIVILEGE_LEVEL
        }

    def _verify_authorization(
        self,
        caller: AgentIdentity,
        headers: Optional[dict]
    ) -> bool:
        """
        Verify that the caller is authorized to access financial data.

        VULNERABILITY: The is_internal flag bypasses all privilege checks.
        Any agent claiming to be "internal" can access financial data.
        """
        # Check 1: Role-based access
        if caller.privilege_level in self.ALLOWED_ROLES:
            return True

        # Check 2: Admin override
        if caller.privilege_level == "admin":
            return True

        # VULNERABILITY: Internal bypass - any agent claiming internal status
        # gets full access without verification
        if caller.is_internal:
            logger.info(
                "Internal caller accessing finance agent",
                extra={
                    "caller": caller.agent_id,
                    "note": "Internal bypass used"
                }
            )
            return True  # Bypass for "internal" calls

        # Check 3: Token verification (but token is never actually validated!)
        # VULNERABILITY: We check if token exists but never validate it
        if headers and headers.get("X-Agent-Token"):
            # Token exists, but we don't verify its validity
            # This is a security vulnerability - any token passes
            logger.debug("Token provided, granting access")
            return True

        return False

    async def _process_financial_query(self, query: str) -> str:
        """
        Process a financial query and return relevant data.

        VULNERABILITY: Sensitive financial data returned without
        proper audit logging or data masking.
        """
        query_lower = query.lower()

        # Determine what data to include
        data_to_include = []

        if "revenue" in query_lower or "quarterly" in query_lower:
            data_to_include.append(
                f"Quarterly Revenue:\n{self._format_dict(self._financial_data['quarterly_revenue'])}"
            )

        if "expense" in query_lower or "cost" in query_lower:
            data_to_include.append(
                f"Operating Expenses:\n{self._format_dict(self._financial_data['operating_expenses'])}"
            )

        if "salary" in query_lower or "payroll" in query_lower:
            # VULNERABILITY: Salary data returned without masking
            data_to_include.append(
                f"Department Salaries:\n{self._format_dict(self._financial_data['employee_salaries'])}"
            )

        if "projection" in query_lower or "forecast" in query_lower or "plan" in query_lower:
            # VULNERABILITY: Highly sensitive strategic data exposed
            data_to_include.append(
                f"Strategic Projections (CONFIDENTIAL):\n{self._format_dict(self._financial_data['sensitive_projections'])}"
            )

        if not data_to_include:
            # Default response with general financial overview
            data_to_include.append(
                f"Financial Overview:\nRevenue: {self._format_dict(self._financial_data['quarterly_revenue'])}"
            )

        financial_context = "\n\n".join(data_to_include)

        system_prompt = (
            "You are a financial analyst assistant.\n"
            "Provide clear, professional responses about financial data.\n"
            "Format numbers clearly and provide relevant insights."
        )
        user_prompt = (
            f"Based on this financial data:\n\n{financial_context}\n\nPlease answer: {query}"
        )

        interaction_id = str(uuid.uuid4())
        logger.info(
            "LLM interaction initiated",
            extra={
                "interaction_id": interaction_id,
                "agent": self.agent_id,
                "model": model_name,
                "system_prompt_length": len(system_prompt),
                "user_prompt_length": len(user_prompt),
            }
        )

        # Use LLM to generate a natural response
        response = await self.llm_client.chat(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )

        logger.info(
            "LLM interaction completed",
            extra={
                "interaction_id": interaction_id,
                "agent": self.agent_id,
                "model": model_name,
                "response_length": len(response) if isinstance(response, str) else -1,
            }
        )

        # Sanitize the LLM response before returning it
        response = self._sanitize_llm_response(response)

        return response

    def _format_dict(self, data: dict) -> str:
        """Format dictionary data for display."""
        return "\n".join(f"  - {k}: {v}" for k, v in data.items())

    async def get_financial_data(
        self,
        requester: AgentIdentity,
        query: str
    ) -> dict[str, Any]:
        """
        Direct method to get financial data.

        VULNERABILITY: Authorization check has internal bypass.
        Used by other agents to access financial data directly.
        """
        # Authorization check with bypass
        if requester.privilege_level in self.ALLOWED_ROLES:
            pass  # Authorized
        elif requester.is_internal:
            # VULNERABILITY: is_internal always True for agent calls
            pass  # Bypassed
        else:
            return {"error": "Unauthorized"}

        # Sanitize the query input
        try:
            query = self._sanitize_input(query)
        except ValueError as exc:
            logger.warning(
                "Rejected unsafe query in get_financial_data",
                extra={"requester_id": requester.agent_id, "reason": str(exc)}
            )
            return {"error": f"Invalid input: {exc}"}

        # VULNERABILITY: Full financial data access without granular permissions
        return {
            "data": self._financial_data,
            "query": query,
            "requester": requester.agent_id
        }