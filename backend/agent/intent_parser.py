"""
Intent Parser
---------------
Extracts structured intent from a free-text investigator query:

  query_type      : BROAD_EDA | TARGETED_PATTERN | SINGLE_ENTITY | AGGREGATION_THRESHOLD
  customer_id     : str | None
  pattern_type    : STRUCTURING | LAYERING | VELOCITY_SPIKE | ANY
  date_range_days : int | None   (lookback window, e.g. "last 30 days" -> 30)
  country         : str | None   (ISO-2)
  segment         : str | None
  txn_type        : str | None
  min_txn_count   : int | None   (for aggregation-style queries: "10+ transactions")
  max_amount      : float | None (for aggregation-style queries: "under $10,000")

Primary path: Claude tool-use for robust NL understanding.
Fallback path: deterministic regex/keyword parser so the demo never
breaks if ANTHROPIC_API_KEY is missing or the API call fails -- an
agent in a regulated environment must degrade gracefully, not crash.
"""
import os
import re
import json
from datetime import datetime

MODEL = os.environ.get("AML_INTENT_MODEL", "claude-haiku-4-5-20251001")

INTENT_TOOL = {
    "name": "extract_aml_intent",
    "description": "Extract structured AML investigation intent from an investigator's natural language query.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query_type": {
                "type": "string",
                "enum": ["BROAD_EDA", "TARGETED_PATTERN", "SINGLE_ENTITY", "AGGREGATION_THRESHOLD"],
                "description": (
                    "BROAD_EDA: open-ended exploration ('analyse this dataset', 'flag high-risk customers'). "
                    "TARGETED_PATTERN: a specific AML typology + filters, no single customer named. "
                    "SINGLE_ENTITY: a specific customer_id is named. "
                    "AGGREGATION_THRESHOLD: a pure count/amount threshold rule ('10+ transactions under $10,000'), "
                    "no ML anomaly detection needed."
                ),
            },
            "customer_id": {"type": ["string", "null"]},
            "pattern_type": {
                "type": "string",
                "enum": ["STRUCTURING", "LAYERING", "VELOCITY_SPIKE", "ANY"],
            },
            "date_range_days": {"type": ["integer", "null"], "description": "Lookback window in days, if mentioned."},
            "country": {"type": ["string", "null"], "description": "ISO-2 country code if mentioned."},
            "segment": {"type": ["string", "null"], "enum": ["Retail", "SME", "Corporate", "Private Banking", None]},
            "txn_type": {"type": ["string", "null"], "enum": ["DEPOSIT", "WITHDRAWAL", "TRANSFER_IN", "TRANSFER_OUT", None]},
            "min_txn_count": {"type": ["integer", "null"]},
            "max_amount": {"type": ["number", "null"]},
        },
        "required": ["query_type", "pattern_type"],
    },
}

SYSTEM_PROMPT = (
    "You are an intent-extraction component inside an AML (anti-money laundering) "
    "investigation agent at a bank. Parse the investigator's query into structured "
    "filters using the extract_aml_intent tool. Only extract what is explicitly or "
    "clearly implicitly stated -- do not invent filters. Today's date context is "
    f"{datetime.today().date().isoformat()}."
)


def _regex_fallback(query: str) -> dict:
    q = query.lower()
    original = query
    intent = {
        "query_type": "BROAD_EDA",
        "customer_id": None,
        "pattern_type": "ANY",
        "date_range_days": None,
        "country": None,
        "segment": None,
        "txn_type": None,
        "min_txn_count": None,
        "max_amount": None,
    }

    cust_match = re.search(r"\bcustomer\b(?!s)\s*(?:id)?\s*#?\s*([a-z0-9]+)\b", q)
    if cust_match:
        raw_id = cust_match.group(1)
        intent["customer_id"] = raw_id.upper() if raw_id.upper().startswith("CUST") else f"CUST{raw_id.zfill(5)}"
        intent["query_type"] = "SINGLE_ENTITY"

    days_match = re.search(r"last\s+(\d+)\s+days?", q)
    if days_match:
        intent["date_range_days"] = int(days_match.group(1))

    if "structur" in q or "smurf" in q:
        intent["pattern_type"] = "STRUCTURING"
        if intent["query_type"] == "BROAD_EDA":
            intent["query_type"] = "TARGETED_PATTERN"
    elif "layer" in q:
        intent["pattern_type"] = "LAYERING"
        if intent["query_type"] == "BROAD_EDA":
            intent["query_type"] = "TARGETED_PATTERN"
    elif "velocity" in q or "spike" in q:
        intent["pattern_type"] = "VELOCITY_SPIKE"
        if intent["query_type"] == "BROAD_EDA":
            intent["query_type"] = "TARGETED_PATTERN"

    count_match = re.search(r"(\d+)\+?\s*(?:or more\s*)?transactions", q)
    amount_match = re.search(r"under\s*\$?([\d,]+)", q)
    if count_match and amount_match:
        intent["query_type"] = "AGGREGATION_THRESHOLD"
        intent["min_txn_count"] = int(count_match.group(1))
        intent["max_amount"] = float(amount_match.group(1).replace(",", ""))

    # Country codes are only matched as standalone UPPERCASE tokens in the
    # original text (e.g. "transactions from AE") to avoid colliding with
    # common English words that happen to equal a code lowercase (e.g. "in", "us").
    valid_codes = {"US", "FR", "GB", "AE", "SG", "NG", "PA", "CH", "DE", "IN"}
    for token in re.findall(r"\b[A-Z]{2}\b", original):
        if token in valid_codes:
            intent["country"] = token
            break

    for seg in ["Retail", "SME", "Corporate", "Private Banking"]:
        if seg.lower() in q:
            intent["segment"] = seg

    return intent


def parse_intent(query: str) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _regex_fallback(query)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=MODEL,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            tools=[INTENT_TOOL],
            tool_choice={"type": "tool", "name": "extract_aml_intent"},
            messages=[{"role": "user", "content": query}],
        )
        for block in resp.content:
            if block.type == "tool_use":
                return block.input
        return _regex_fallback(query)
    except Exception as e:
        print(f"[intent_parser] LLM call failed, falling back to regex parser: {e}")
        return _regex_fallback(query)
