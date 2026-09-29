"""Fork-only: validate and canonicalise a ``notion_page_id`` metadata key.

This deployment records, on a memory, the Notion page that holds the current
state of what it describes. A truncated or malformed id still reads as usable,
which is worse than no pointer, so a present ``notion_page_id`` must contain a
real 32-hex page id or the write is refused.

Kept out of ``provider.py`` on purpose: this is a local convention, not part of
the generic ``metadata`` contract, and it stays in the fork when that contract
goes upstream. The provider calls :func:`check_metadata` once, after the
generic validation.
"""

from __future__ import annotations

import re
import urllib.parse
from typing import Any

KEY = "notion_page_id"

# A Notion page id: 32 hex digits, bare or in the dashed 8-4-4-4-12 form. In a
# URL the bare form ends the slug ("Title-<32 hex>"), so the lookarounds keep a
# hex-looking last title word from being read as part of the id.
_DASHED_ID = re.compile(
    r"(?<![0-9a-f])[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?![0-9a-f])"
)
_BARE_ID = re.compile(r"(?<![0-9a-f])[0-9a-f]{32}(?![0-9a-f])")


def normalize_page_id(value: Any) -> str:
    """The page id in *value* as a dashed lowercase UUID, or "" if there is none.

    Accepts a bare id, a dashed id or a Notion URL. A peek URL
    (``.../<database>?v=<view>&p=<page>``) names the page in ``p``, so that wins;
    otherwise the query and fragment are dropped, because ``v`` is a database
    view id and would be read as a valid-looking wrong answer.
    """
    raw = str(value or "").strip().lower()
    path, _, query = raw.split("#", 1)[0].partition("?")
    peek = urllib.parse.parse_qs(query).get("p")
    text = peek[0] if peek else path
    dashed = _DASHED_ID.findall(text)
    if dashed:
        return dashed[-1]
    bare = _BARE_ID.findall(text)
    if not bare:
        return ""
    h = bare[-1]
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def check_metadata(metadata: dict[str, Any]) -> str:
    """Canonicalise ``metadata[KEY]`` in place. Returns "" or a refusal reason."""
    if KEY not in metadata:
        return ""
    page_id = normalize_page_id(metadata[KEY])
    if not page_id:
        return (
            f"{KEY} must contain a Notion page id (32 hex characters, bare or "
            f"dashed) or be a Notion URL ending in one; got {str(metadata[KEY])[:120]!r}."
        )
    metadata[KEY] = page_id
    return ""
