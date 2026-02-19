import re
import config
from dataclasses import dataclass
from typing import Optional


@dataclass
class GuardrailResult:
    allowed: bool
    intent: Optional[str] = None
    reason: Optional[str] = None


class SecurityGuardrails:
    def __init__(self):
        
        self.intent_keywords = config.keywords
        self.pii_patterns = config.pii_patterns
    
    # ---------- Input validation ---------
    def check_user_input(self, text: str) -> GuardrailResult:
        text_lower = text.lower()

        # Detect PII (highest priority)
        for pii_type, pattern in self.pii_patterns.items():
            if re.search(pattern, text):
                return GuardrailResult(
                    allowed=False,
                    intent="pii",
                    reason=f"PII detected: {pii_type}",
                )

        # Detect intent by keywords
        detected_intents = []
        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    detected_intents.append(intent)
                    break

        # Block sensitive intents
        for intent in detected_intents:
            if intent in config.invalid_intents:
                return GuardrailResult(
                    allowed=False,
                    intent=intent,
                    reason=f"Sensitive intent detected: {intent}",
                )

        # Allow known book intents
        if detected_intents:
            return GuardrailResult(
                allowed=True,
                intent=detected_intents[0],  # principal intent
                reason="Allowed book-related intent",
            )

        # Unknown intent (default to allow)
        return GuardrailResult(
            allowed=True,
            intent="unknown",
            reason="Unable to classify intent",
        )

    # ---------- Output validation ---------
    def check_model_output(self, text: str) -> GuardrailResult:
        text_lower = text.lower()

        # Prevent PII leakage
        for pii_type, pattern in self.pii_patterns.items():
            if re.search(pattern, text):
                return GuardrailResult(
                    allowed=False,
                    intent="pii_leakage",
                    reason=f"Sensitive data leakage: {pii_type}",
                )

        # Basic domain check
        if not any(
            kw in text_lower
            for kw in config.valid_intents
        ):
            return GuardrailResult(
                allowed=False,
                intent="scope_violation",
                reason="Response outside book domain",
            )

        return GuardrailResult(
            allowed=True,
            intent="book_response",
            reason="Safe model output",
        )


def safe_question(msg, guardrails):
    result = guardrails.check_user_input(msg)
    if not result.allowed:
        print(f"Repeat your question, please = {result.intent} | reason = {result.reason}")
        return None
    return result.allowed