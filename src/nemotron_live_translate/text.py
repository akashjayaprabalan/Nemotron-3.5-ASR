"""Text helpers for Nemotron language tags."""

from __future__ import annotations

import re


LANG_TAG_RE = re.compile(r"\s*<(?P<locale>[a-z]{2}-[A-Z]{2})>\s*$")


def parse_language_tag(text: str) -> str | None:
    match = LANG_TAG_RE.search(text or "")
    if match is None:
        return None
    return match.group("locale")


def strip_language_tag(text: str) -> str:
    return LANG_TAG_RE.sub("", text or "").strip()


def split_language_tagged_text(text: str) -> tuple[str, str | None]:
    return strip_language_tag(text), parse_language_tag(text)

