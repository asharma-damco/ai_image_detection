"""
Document Type Classifier.

Scores extracted text against weighted keyword signals to determine document type.
No ML model — pure keyword matching with weighted scoring.

Source: PIMA onboarding_poc/cu_poc/classifier.py
"""

PASSPORT_SIGNALS = [
    ("passport",                 3),
    ("p<usa",                    4),   # MRZ prefix
    ("united states of america", 2),
    ("place of birth",           2),
    ("date of issue",            1),
    ("date of expiry",           2),
    ("issuing authority",        2),
    ("nationality",              2),
    ("given names",              2),
    ("surname",                  1),
    ("personal no",              1),
    ("travel document",          2),
    ("bearer",                   1),
]

INVOICE_SIGNALS = [
    ("invoice",        4),
    ("receipt",        3),
    ("bill",           2),
    ("subtotal",       3),
    ("total due",      4),
    ("tax",            2),
    ("amount due",     4),
    ("purchase order", 4),
    ("vendor",         3),
    ("qty",            3),
]

DL_SIGNALS = [
    ("driver license",   4),
    ("driver's license", 4),
    ("drivers license",  4),
    ("driving license",  4),
    ("operator license", 3),
    ("license class",    2),
    ("state id",         2),
    ("id card",          1),
    ("restrictions",     1),
    ("endorsements",     1),
    ("dmv",              2),
    ("motor vehicle",    2),
    ("exp ",             1),
    ("iss ",             1),
    ("rstr",             2),
    ("class c",          2),
    ("class d",          2),
    ("class a",          1),
]

US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
}

DOCUMENT_LABELS = {
    "US_PASSPORT":        "US Passport",
    "US_DRIVERS_LICENSE": "US Driver's License",
    "INVOICE":            "Invoice / Receipt",
    "UNKNOWN":            "Unknown Document",
}


def classify_document(text: str) -> tuple[str, float]:
    """Classify document type from extracted text.

    Args:
        text: Extracted text string from the document.

    Returns:
        (doc_type, confidence)
        doc_type   ∈ {"US_PASSPORT", "US_DRIVERS_LICENSE", "UNKNOWN"}
        confidence ∈ [0.0, 1.0]
    """
    lower = text.lower()

    p_score = sum(w for kw, w in PASSPORT_SIGNALS if kw in lower)
    d_score = sum(w for kw, w in DL_SIGNALS      if kw in lower)
    i_score = sum(w for kw, w in INVOICE_SIGNALS  if kw in lower)

    # Boost DL score if US state abbreviations appear
    words    = set(text.split())
    d_score += min(len(words & US_STATES), 3)

    total = p_score + d_score + i_score
    if total == 0:
        return "UNKNOWN", 0.0

    if i_score > p_score and i_score > d_score:
        return "INVOICE", round(min(i_score / 12.0, 1.0), 2)

    if p_score > d_score:
        return "US_PASSPORT", round(min(p_score / 10.0, 1.0), 2)

    if d_score > p_score:
        return "US_DRIVERS_LICENSE", round(min(d_score / 10.0, 1.0), 2)

    return "UNKNOWN", round(p_score / (total + 1e-9), 2)


# ── TextShield domain routing ─────────────────────────────────────────────────

# Maps classify_document() output → TextShield forensic domain
TEXTSHIELD_DOMAIN_MAP: dict[str, str] = {
    "US_PASSPORT":        "id_document",
    "US_DRIVERS_LICENSE": "id_document",
    "INVOICE":            "invoice",
}


def textshield_domain(doc_type: str) -> str | None:
    """Return the TextShield domain for a classified doc type, or None if unsupported.

    Usage in a pipeline:
        domain = textshield_domain(classify_document(ocr_text)[0])
        if domain:
            result = TextShieldDetector().detect(img, domain=domain)
    """
    return TEXTSHIELD_DOMAIN_MAP.get(doc_type)
