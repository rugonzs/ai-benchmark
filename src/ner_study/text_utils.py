"""Sentence reconstruction and token/offset helpers."""

from __future__ import annotations

from dataclasses import dataclass


OPENING_TOKENS = {"(", "[", "{", "``", '"'}
CLOSING_TOKENS = {
    ".",
    ",",
    ":",
    ";",
    "!",
    "?",
    "%",
    ")",
    "]",
    "}",
    "''",
    '"',
    "'s",
    "'m",
    "'re",
    "'ve",
    "'d",
    "'ll",
    "n't",
}


@dataclass(slots=True)
class TokenOffset:
    token: str
    start: int
    end: int


def normalize_detokenized_token(token: str) -> str:
    replacements = {
        "``": '"',
        "''": '"',
        "-LRB-": "(",
        "-RRB-": ")",
        "-LSB-": "[",
        "-RSB-": "]",
        "-LCB-": "{",
        "-RCB-": "}",
    }
    return replacements.get(token, token)


def should_prepend_space(previous: str | None, token: str) -> bool:
    if previous is None:
        return False
    if token in CLOSING_TOKENS:
        return False
    if previous in OPENING_TOKENS:
        return False
    if token.startswith("'") and token not in {"'"}:
        return False
    if token == "-" or previous == "-":
        return False
    return True


def detokenize_with_offsets(tokens: list[str]) -> tuple[str, list[TokenOffset]]:
    """Reconstruct a sentence and preserve char offsets for each token."""

    pieces: list[str] = []
    offsets: list[TokenOffset] = []
    cursor = 0
    previous: str | None = None

    for raw_token in tokens:
        token = normalize_detokenized_token(raw_token)
        if should_prepend_space(previous, token):
            pieces.append(" ")
            cursor += 1
        start = cursor
        pieces.append(token)
        cursor += len(token)
        offsets.append(TokenOffset(token=token, start=start, end=cursor))
        previous = token

    return "".join(pieces), offsets

