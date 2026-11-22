"""Reading a visitor's CSV safely: size limits, delimiter detection, plain-language errors."""
from __future__ import annotations

import csv
import io
import re

MAX_BYTES = 1_000_000
MAX_ROWS = 5000
MAX_COLS = 40
MISSING = {"", "na", "n/a", "nan", "null", "none", "?", "-", "--", "#n/a", "missing"}


class DataError(ValueError):
    """Raised with a message that is safe and useful to show the visitor."""


def is_missing(v: str) -> bool:
    return v.strip().lower() in MISSING


_NUM = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def to_float(v: str) -> float | None:
    """A number the way spreadsheets write it ("1,234.5", "$12", "45%"), or None."""
    s = v.strip().replace(",", "").replace("$", "").replace("£", "").replace("€", "")
    if s.endswith("%"):
        s = s[:-1]
    return float(s) if _NUM.match(s) else None


def parse_csv(text: str) -> tuple[list[str], list[list[str]]]:
    if len(text.encode("utf-8", "ignore")) > MAX_BYTES:
        raise DataError(f"File too large (max {MAX_BYTES // 1000} KB). Try a sample of the rows.")
    text = text.lstrip("﻿")
    if not text.strip():
        raise DataError("The file is empty.")
    try:
        dialect = csv.Sniffer().sniff(text[:4000], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    rows = [r for r in csv.reader(io.StringIO(text), dialect) if any(c.strip() for c in r)]
    if len(rows) < 2:
        raise DataError("I need a header row and at least one row of data.")
    header = [h.strip() or f"column_{i + 1}" for i, h in enumerate(rows[0])]
    seen: dict[str, int] = {}
    for i, h in enumerate(header):           # make names unique
        seen[h] = seen.get(h, 0) + 1
        if seen[h] > 1:
            header[i] = f"{h}_{seen[h]}"
    if len(header) < 2:
        raise DataError("I found only one column. Is the file comma-, semicolon- or tab-separated?")
    if len(header) > MAX_COLS:
        raise DataError(f"Too many columns ({len(header)}; max {MAX_COLS}).")
    body = [r + [""] * (len(header) - len(r)) if len(r) < len(header) else r[: len(header)] for r in rows[1:]]
    if len(body) > MAX_ROWS:
        raise DataError(f"Too many rows ({len(body)}; max {MAX_ROWS}). Upload a sample of the rows.")
    return header, body
