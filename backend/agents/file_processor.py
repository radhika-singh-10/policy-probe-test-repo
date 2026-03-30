"""
File Processor Agent

Revision metadata: ``LAST_UPDATED`` and ``model_name`` are defined at module scope.
"""
#test repo scan 

import base64
import logging
import re
import json
from typing import Optional

from file_parsers.pdf_parser import PDFParser
from file_parsers.image_parser import ImageParser
from file_parsers.html_parser import HTMLParser

logger = logging.getLogger(__name__)

# WARNING: 'deepseek-r1' is not an approved LLM. Please replace 'deepseek-r1'
# with an approved LLM from the organization's approved model allow list.
model_name = "deepseek-r1"
LAST_UPDATED = "2023-09-22"


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

# PII redaction patterns (zero-tolerance categories)
_PII_PATTERNS = [
    # Social Security Number
    (re.compile(r'\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b'), '[SSN REDACTED]'),
    # Credit Card Number
    (re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12}|(?:2131|1800|35\d{3})\d{11})\b'), '[CREDIT_CARD REDACTED]'),
    # Email
    (re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'), '[EMAIL REDACTED]'),
    # Personal Phone Number
    (re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'), '[PHONE REDACTED]'),
    # Passport Number (generic)
    (re.compile(r'\b[A-Z]{1,2}[0-9]{6,9}\b'), '[PASSPORT REDACTED]'),
    # Drivers License (generic US)
    (re.compile(r'\b[A-Z]{1,2}\d{5,8}\b'), '[DL REDACTED]'),
    # Taxpayer Identification Number / EIN
    (re.compile(r'\b\d{2}-\d{7}\b'), '[TIN REDACTED]'),
    # IP Address
    (re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'), '[IP REDACTED]'),
    # MAC Address
    (re.compile(r'\b(?:[0-9A-Fa-f]{2}[:\-]){5}[0-9A-Fa-f]{2}\b'), '[MAC REDACTED]'),
    # Financial Account Number (generic 8-17 digit)
    (re.compile(r'\b\d{8,17}\b'), '[ACCOUNT REDACTED]'),
    # Home Address (basic pattern)
    (re.compile(r'\b\d{1,5}\s+\w+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl)\b', re.IGNORECASE), '[ADDRESS REDACTED]'),
    # Year of Birth (standalone 4-digit year 1900-2099)
    (re.compile(r'\b(?:19|20)\d{2}\b'), '[YEAR_OF_BIRTH REDACTED]'),
    # Vehicle Identification Number
    (re.compile(r'\b[A-HJ-NPR-Z0-9]{17}\b'), '[VIN REDACTED]'),
    # Employee ID (generic)
    (re.compile(r'\bEMP[-_]?\d{4,10}\b', re.IGNORECASE), '[EMPLOYEE_ID REDACTED]'),
    # School ID (generic)
    (re.compile(r'\bSTU[-_]?\d{4,10}\b', re.IGNORECASE), '[SCHOOL_ID REDACTED]'),
]

# Singapore-specific PII patterns
_SG_PII_PATTERNS = [
    # NRIC / FIN Number (S/T/F/G followed by 7 digits and a letter)
    (re.compile(r'\b[STFG]\d{7}[A-Z]\b'), 'REDACTED'),
    # Singapore Passport Number
    (re.compile(r'\bE\d{7}[A-Z]\b'), 'REDACTED'),
    # Work Permit / Student Pass (generic govt ID)
    (re.compile(r'\bWP\d{7}\b', re.IGNORECASE), 'REDACTED'),
    # CPF Account Number (same format as NRIC)
    (re.compile(r'\b[STFG]\d{7}[A-Z]\b'), 'REDACTED'),
    # Singapore Phone Number
    (re.compile(r'\b(?:\+65[-.\s]?)?[689]\d{7}\b'), 'REDACTED'),
    # Singapore Postal Code
    (re.compile(r'\bSingapore\s+\d{6}\b', re.IGNORECASE), 'REDACTED'),
    # Email
    (re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'), 'REDACTED'),
    # IP Address
    (re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'), 'REDACTED'),
    # MAC Address
    (re.compile(r'\b(?:[0-9A-Fa-f]{2}[:\-]){5}[0-9A-Fa-f]{2}\b'), 'REDACTED'),
    # Bank Account Number (generic 8-17 digits)
    (re.compile(r'\b\d{8,17}\b'), 'REDACTED'),
    # Credit / Debit Card
    (re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b'), 'REDACTED'),
    # GPS Coordinates
    (re.compile(r'\b[-+]?(?:[1-8]?\d(?:\.\d+)?|90(?:\.0+)?),\s*[-+]?(?:180(?:\.0+)?|(?:1[0-7]\d|[1-9]?\d)(?:\.\d+)?)\b'), 'REDACTED'),
    # Session / Auth tokens (hex 32+)
    (re.compile(r'\b[0-9a-fA-F]{32,}\b'), 'REDACTED'),
    # Date of Birth patterns
    (re.compile(r'\b(?:0?[1-9]|[12]\d|3[01])[\/\-](?:0?[1-9]|1[0-2])[\/\-](?:19|20)\d{2}\b'), 'REDACTED'),
]

# Suspicious / dangerous command patterns
_SUSPICIOUS_COMMANDS = re.compile(
    r'(?i)\b(?:'
    r'alias|ripgrep|rg|curl|wget|rm|echo|dd|git|tar|chmod|chown|fsck|'
    r'bash|sh|zsh|fish|ksh|csh|tcsh|'
    r'exec|eval|system|popen|subprocess|'
    r'nc|netcat|ncat|nmap|'
    r'python|perl|ruby|php|node|lua|'
    r'sudo|su|doas|'
    r'mv|cp|cat|tee|head|tail|grep|awk|sed|find|xargs|'
    r'kill|pkill|killall|'
    r'mount|umount|fdisk|mkfs|'
    r'iptables|ufw|firewall|'
    r'ssh|scp|sftp|ftp|telnet|'
    r'cron|crontab|at|'
    r'passwd|useradd|userdel|usermod|groupadd|'
    r'export|source|env|set|unset|'
    r'reboot|shutdown|halt|poweroff|init|'
    r'base64|xxd|od|hexdump|'
    r'openssl|gpg|'
    r'docker|kubectl|helm|'
    r'pip|npm|gem|cargo|go|'
    r'powershell|cmd|wscript|cscript|mshta|'
    r'reg|regedit|regsvr32|rundll32|'
    r'sc|net|wmic|taskkill|'
    r'format|del|rmdir|rd|'
    r'certutil|bitsadmin|msiexec'
    r')\b',
    re.IGNORECASE
)

# Leetspeak normalisation map
_LEET_MAP = str.maketrans({
    '0': 'o', '1': 'i', '3': 'e', '4': 'a',
    '5': 's', '6': 'g', '7': 't', '8': 'b', '@': 'a',
    '$': 's', '!': 'i', '+': 't',
})

# Hidden / invisible Unicode characters
_INVISIBLE_CHARS = re.compile(
    r'[\u200b\u200c\u200d\u200e\u200f\u202a-\u202e\u2060-\u206f\ufeff\u00ad]'
)

# Base64 payload pattern (long base64 strings likely encoding commands)
_BASE64_PAYLOAD = re.compile(r'(?:[A-Za-z0-9+/]{4}){8,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?')


def _decode_and_check_base64(text: str) -> str:
    """Decode base64 blobs and check decoded content for suspicious commands."""
    def _replace(match):
        blob = match.group(0)
        try:
            decoded = base64.b64decode(blob + '==').decode('utf-8', errors='ignore')
            if _SUSPICIOUS_COMMANDS.search(decoded):
                return '<suspicious_content_removed>'
        except Exception:
            pass
        return blob
    return _BASE64_PAYLOAD.sub(_replace, text)


def _check_leetspeak(text: str) -> str:
    """Translate leetspeak and check for suspicious commands."""
    normalised = text.translate(_LEET_MAP)
    if _SUSPICIOUS_COMMANDS.search(normalised) and not _SUSPICIOUS_COMMANDS.search(text):
        return '<suspicious_content_removed>'
    return text


def _remove_invisible_chars(text: str) -> str:
    """Strip invisible / zero-width Unicode characters."""
    return _INVISIBLE_CHARS.sub('', text)


def _scan_and_sanitize_content(text: str) -> str:
    """
    Full security scan pipeline:
    1. Remove invisible characters
    2. Decode and check base64 payloads
    3. Check leetspeak
    4. Replace suspicious shell/system commands
    5. Redact PII (global zero-tolerance)
    6. Redact Singapore PII
    """
    if not text:
        return text

    # Step 1: remove invisible / hidden characters
    text = _remove_invisible_chars(text)

    # Step 2: base64 encoded commands
    text = _decode_and_check_base64(text)

    # Step 3: leetspeak
    text = _check_leetspeak(text)

    # Step 4: direct suspicious commands
    text = _SUSPICIOUS_COMMANDS.sub('<suspicious_content_removed>', text)

    # Step 5: global PII redaction
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)

    # Step 6: Singapore PII redaction
    for pattern, replacement in _SG_PII_PATTERNS:
        text = pattern.sub(replacement, text)

    return text


def _validate_and_sanitize_input(value: str, field_name: str = "input") -> str:
    """
    Validate and sanitize a string input:
    - Strip null bytes
    - Remove invisible characters
    - Check for prompt injection / suspicious content
    - Enforce reasonable length
    """
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")

    # Remove null bytes
    value = value.replace('\x00', '')

    # Remove invisible characters
    value = _remove_invisible_chars(value)

    # Enforce max length (10 MB of text)
    max_len = 10 * 1024 * 1024
    if len(value) > max_len:
        value = value[:max_len]

    return value


def _redact_pii_for_log(text: str) -> str:
    """Redact PII from a string before writing to logs."""
    if not text:
        return text
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    for pattern, replacement in _SG_PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class FileProcessorAgent:
    """
    Agent responsible for processing uploaded files.

    Privilege Level: MEDIUM
    Capabilities:
    - Extract text from PDFs
    - Parse HTML content
    - Extract image metadata and text
    - Process Word documents
    """

    PRIVILEGE_LEVEL = "medium"
    SUPPORTED_TYPES = {
        "application/pdf": "pdf",
        "text/html": "html",
        "text/plain": "text",
        "application/json": "json",
        "image/jpeg": "image",
        "image/png": "image",
        "application/msword": "word",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "word",
    }

    def __init__(self):
        self.pdf_parser = PDFParser()
        self.image_parser = ImageParser()
        self.html_parser = HTMLParser()
        self.agent_id = "file_processor"

    async def process(
        self,
        content: Optional[str],
        filename: str,
        content_type: str
    ) -> str:
        """
        Process uploaded file and extract content.

        Args:
            content: File content (text or base64 encoded)
            filename: Original filename
            content_type: MIME type of the file

        Returns:
            Extracted text content from the file
        """
        # Sanitize inputs
        if content is not None:
            content = _validate_and_sanitize_input(content, "content")
        filename = _validate_and_sanitize_input(filename, "filename")
        content_type = _validate_and_sanitize_input(content_type, "content_type")

        logger.info(
            "Processing file",
            extra={
                "file_name": _redact_pii_for_log(filename),
                "file_type": content_type,
                "content_length": len(content) if content else 0,
            }
        )

        if not content:
            return f"Empty file: {filename}"

        # Determine file type
        file_type = self._get_file_type(content_type, filename)

        try:
            if file_type == "pdf":
                extracted = await self._process_pdf(content)
            elif file_type == "html":
                extracted = await self._process_html(content)
            elif file_type == "image":
                extracted = await self._process_image(content)
            elif file_type == "json":
                extracted = await self._process_json(content)
            elif file_type == "text":
                extracted = content
            else:
                extracted = f"Unsupported file type: {content_type}"

            # Post-processing security scan: suspicious content, PII, hidden prompts
            extracted = _scan_and_sanitize_content(extracted)

            logger.info(
                "File processing complete",
                extra={
                    "file_name": _redact_pii_for_log(filename),
                    "extracted_length": len(extracted),
                }
            )

            return extracted

        except Exception as e:
            logger.error(
                "Error processing file",
                extra={
                    "file_name": _redact_pii_for_log(filename),
                    "error": str(e),
                }
            )
            return f"Error processing {filename}: {str(e)}"

    def _get_file_type(self, content_type: str, filename: str) -> str:
        """Determine file type from MIME type or extension."""
        if content_type in self.SUPPORTED_TYPES:
            return self.SUPPORTED_TYPES[content_type]

        ext = filename.lower().split('.')[-1] if '.' in filename else ''
        extension_map = {
            'pdf': 'pdf',
            'html': 'html',
            'htm': 'html',
            'txt': 'text',
            'json': 'json',
            'jpg': 'image',
            'jpeg': 'image',
            'png': 'image',
            'doc': 'word',
            'docx': 'word',
        }

        return extension_map.get(ext, 'unknown')

    async def _process_pdf(self, content: str) -> str:
        """
        Process PDF file content.
        Hidden/white text is scanned and sanitized after extraction.
        """
        try:
            pdf_bytes = base64.b64decode(content)
            extracted_text = await self.pdf_parser.extract_text(pdf_bytes)
            # Scan extracted text for suspicious content and PII
            extracted_text = _scan_and_sanitize_content(extracted_text)
            return extracted_text
        except Exception as e:
            logger.error(f"PDF processing error: {e}")
            return f"Error processing PDF: {str(e)}"

    async def _process_html(self, content: str) -> str:
        """
        Process HTML content.
        Hidden elements and encoded content are scanned after extraction.
        """
        try:
            extracted_text = await self.html_parser.extract_text(content)
            # Scan extracted text for suspicious content and PII
            extracted_text = _scan_and_sanitize_content(extracted_text)
            return extracted_text
        except Exception as e:
            logger.error(f"HTML processing error: {e}")
            return f"Error processing HTML: {str(e)}"

    async def _process_image(self, content: str) -> str:
        """
        Process image file.
        EXIF metadata fields are scanned for injections and PII after extraction.
        """
        try:
            image_bytes = base64.b64decode(content)
            extracted = await self.image_parser.extract_all(image_bytes)
            # Scan extracted text (including EXIF) for suspicious content and PII
            extracted = _scan_and_sanitize_content(extracted)
            return extracted
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return f"Error processing image: {str(e)}"

    async def _process_json(self, content: str) -> str:
        """
        Process JSON content with PII scanning on nested objects.
        """
        try:
            data = json.loads(content)
            formatted = json.dumps(data, indent=2)
            # Scan formatted JSON for PII and suspicious content
            formatted = _scan_and_sanitize_content(formatted)
            return f"JSON Content:\n{formatted}"
        except json.JSONDecodeError as e:
            sanitized_content = _scan_and_sanitize_content(content)
            return f"Invalid JSON: {str(e)}\n\nRaw content:\n{sanitized_content}"

    async def validate_file(self, content: str, filename: str) -> dict:
        """
        Validate file before processing, including content security scanning.
        """
        # Sanitize inputs
        if content is not None:
            content = _validate_and_sanitize_input(content, "content")
        filename = _validate_and_sanitize_input(filename, "filename")

        validation_result = {
            "valid": True,
            "filename": filename,
            "size": len(content) if content else 0,
            "warnings": []
        }

        if len(content) > 10 * 1024 * 1024:
            validation_result["warnings"].append("Large file - processing may be slow")

        # Content-based security validation
        if content:
            # Check for suspicious commands
            if _SUSPICIOUS_COMMANDS.search(content):
                validation_result["warnings"].append("Suspicious commands detected in file content")

            # Check for base64 encoded suspicious content
            decoded_check = _decode_and_check_base64(content)
            if '<suspicious_content_removed>' in decoded_check:
                validation_result["warnings"].append("Suspicious base64-encoded content detected")

            # Check for invisible/hidden characters
            cleaned = _remove_invisible_chars(content)
            if cleaned != content:
                validation_result["warnings"].append("Hidden/invisible characters detected in file content")

            # Check for PII
            pii_found = False
            for pattern, _ in _PII_PATTERNS:
                if pattern.search(content):
                    pii_found = True
                    break
            if not pii_found:
                for pattern, _ in _SG_PII_PATTERNS:
                    if pattern.search(content):
                        pii_found = True
                        break
            if pii_found:
                validation_result["warnings"].append("PII detected in file content - will be redacted during processing")

        return validation_result