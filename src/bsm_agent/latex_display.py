"""Helpers for rendering long LaTeX display equations."""

from __future__ import annotations

import re

_ALLOWBREAK_TOKEN = r" \allowbreak "
_MAX_PIECES_PER_DISPLAY = 28
_WRAP_PATTERNS = (
    re.compile(
        r"^(?P<prefix>.*?)\\bigl\(\s*(?P<body>.*)\s*\\bigr\)(?P<suffix>\s*(?:\+\s*\\mathrm\{h\.c\.\})?)$",
        re.DOTALL,
    ),
    re.compile(
        r"^(?P<prefix>.*?)\\left\(\s*(?P<body>.*)\s*\\right\)(?P<suffix>\s*(?:\+\s*\\mathrm\{h\.c\.\})?)$",
        re.DOTALL,
    ),
)


def _wrap_dmath(term: str) -> str:
    return f"""
\\begin{{dmath*}}
{term}
\\end{{dmath*}}
"""


def _chunk_allowbreak_term(term: str) -> list[str] | None:
    if _ALLOWBREAK_TOKEN not in term:
        return None
    for pattern in _WRAP_PATTERNS:
        match = pattern.match(term.strip())
        if match is None:
            continue
        prefix = match.group("prefix").strip()
        suffix = match.group("suffix").strip()
        pieces = [piece.strip() for piece in match.group("body").split(_ALLOWBREAK_TOKEN) if piece.strip()]
        if len(pieces) <= _MAX_PIECES_PER_DISPLAY:
            return None
        chunks = [
            _ALLOWBREAK_TOKEN.join(pieces[index : index + _MAX_PIECES_PER_DISPLAY])
            for index in range(0, len(pieces), _MAX_PIECES_PER_DISPLAY)
        ]
        rendered: list[str] = []
        rendered.append(_wrap_dmath(rf"{prefix}\Biggl\{{ {chunks[0]}"))
        for chunk in chunks[1:-1]:
            rendered.append(_wrap_dmath(rf"\qquad {chunk}"))
        closing = rf"\qquad {chunks[-1]} \Biggr\}}"
        if suffix:
            closing += " " + suffix
        rendered.append(_wrap_dmath(closing))
        return rendered
    return None


def render_display_equations(body: str, *, separator: str) -> str:
    if not body.strip():
        return r"""\[
\text{No terms generated.}
\]"""
    terms = [term.strip() for term in body.split(separator) if term.strip() and term.strip() != "0"]
    rendered: list[str] = []
    for term in terms:
        chunks = _chunk_allowbreak_term(term)
        if chunks is not None:
            rendered.extend(chunks)
        else:
            rendered.append(_wrap_dmath(term))
    return "\n".join(rendered)
