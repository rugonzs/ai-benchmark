"""Conversions between IOB tag sequences, spans, and strict JSON outputs."""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from ner_study.schemas import Span
from ner_study.text_utils import TokenOffset


def iob_tags_to_spans(tokens: list[str], tags: list[str], offsets: list[TokenOffset]) -> list[Span]:
    spans: list[Span] = []
    current_label: str | None = None
    current_start_token: int | None = None

    def flush(end_token_index: int | None) -> None:
        nonlocal current_label, current_start_token
        if current_label is None or current_start_token is None or end_token_index is None:
            current_label = None
            current_start_token = None
            return
        start_offset = offsets[current_start_token]
        end_offset = offsets[end_token_index]
        text = "".join(offset.token if i == current_start_token else offset.token for i, offset in enumerate(offsets[current_start_token : end_token_index + 1], start=current_start_token))
        text = sentence_slice_from_offsets(offsets, current_start_token, end_token_index)
        spans.append(
            Span(
                start=start_offset.start,
                end=end_offset.end,
                text=text,
                label=current_label,
            )
        )
        current_label = None
        current_start_token = None

    for index, tag in enumerate(tags):
        if tag == "O":
            flush(index - 1 if current_label is not None else None)
            continue

        prefix, label = tag.split("-", 1)
        if prefix == "B" or current_label != label:
            flush(index - 1 if current_label is not None else None)
            current_label = label
            current_start_token = index
            continue

    flush(len(tags) - 1 if current_label is not None else None)
    return spans


def sentence_slice_from_offsets(offsets: list[TokenOffset], start_idx: int, end_idx: int) -> str:
    start = offsets[start_idx].start
    end = offsets[end_idx].end
    pieces: list[str] = []
    cursor = start
    for offset in offsets[start_idx : end_idx + 1]:
        if offset.start > cursor:
            pieces.append(" " * (offset.start - cursor))
        pieces.append(offset.token)
        cursor = offset.end
    return "".join(pieces)


def spans_to_json_text(spans: list[Span]) -> str:
    return json.dumps([span.model_dump() for span in spans], ensure_ascii=True, separators=(",", ":"))


def spans_to_text_label_json(spans: list[Span]) -> str:
    payload = [{"text": span.text, "label": span.label} for span in spans]
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def parse_json_spans(raw_text: str, sentence: str, repair_text_offsets: bool = False) -> tuple[list[Span], list[str]]:
    errors: list[str] = []
    raw_text = extract_json_candidate(raw_text)
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return [], [f"Invalid JSON: {exc}"]

    if not isinstance(payload, list):
        return [], ["Top-level payload must be a JSON list."]

    spans: list[Span] = []
    for index, item in enumerate(payload):
        try:
            span = Span.model_validate(item)
        except ValidationError as exc:
            errors.append(f"Item {index}: {exc.errors()}")
            continue
        if span.end > len(sentence):
            errors.append(f"Item {index}: span end {span.end} exceeds sentence length {len(sentence)}.")
            continue
        if sentence[span.start : span.end] != span.text:
            if repair_text_offsets:
                repaired = repair_span_offsets(span, sentence)
                if repaired is not None:
                    errors.append(
                        f"Item {index}: repaired offsets from ({span.start}, {span.end}) "
                        f"to ({repaired.start}, {repaired.end}) for text {span.text!r}."
                    )
                    spans.append(repaired)
                    continue
            errors.append(
                f"Item {index}: text mismatch. Expected {sentence[span.start:span.end]!r}, received {span.text!r}."
            )
            continue
        spans.append(span)

    overlaps = validate_non_overlapping_spans(spans)
    errors.extend(overlaps)
    return sorted(spans, key=lambda item: (item.start, item.end, item.label)), errors


def parse_json_text_label_spans(raw_text: str, sentence: str) -> tuple[list[Span], list[str]]:
    """Parse [{"text": "...", "label": "..."}] and map text mentions to character spans."""

    errors: list[str] = []
    raw_text = extract_json_candidate(raw_text)
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return [], [f"Invalid JSON: {exc}"]
    if not isinstance(payload, list):
        return [], ["Top-level payload must be a JSON list."]

    spans: list[Span] = []
    used_ranges: set[tuple[int, int]] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            errors.append(f"Item {index}: expected object with text and label.")
            continue
        text = item.get("text")
        label = item.get("label")
        if not isinstance(text, str) or not isinstance(label, str):
            errors.append(f"Item {index}: text and label must be strings.")
            continue
        if not text:
            errors.append(f"Item {index}: empty text.")
            continue
        matches = find_text_matches(sentence, text)
        if not matches:
            errors.append(f"Item {index}: text {text!r} not found in sentence.")
            continue
        chosen = next((match for match in matches if match not in used_ranges), matches[0])
        used_ranges.add(chosen)
        try:
            spans.append(Span(start=chosen[0], end=chosen[1], text=text, label=label))
        except Exception as exc:
            errors.append(f"Item {index}: {exc}")

    errors.extend(validate_non_overlapping_spans(spans))
    return sorted(spans, key=lambda item: (item.start, item.end, item.label)), errors


def extract_json_candidate(raw_text: str) -> str:
    stripped = raw_text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    start_positions = [pos for pos in (stripped.find("["), stripped.find("{")) if pos != -1]
    if not start_positions:
        return stripped
    start = min(start_positions)
    return stripped[start:].strip()


def repair_span_offsets(span: Span, sentence: str) -> Span | None:
    """Repair offsets when the model got the text right but character indexes wrong."""

    if not span.text:
        return None
    matches = find_text_matches(sentence, span.text)
    if not matches:
        return None
    supplied_center = (span.start + span.end) / 2
    start, end = min(matches, key=lambda item: abs(((item[0] + item[1]) / 2) - supplied_center))
    return Span(start=start, end=end, text=span.text, label=span.label)


def find_text_matches(sentence: str, text: str) -> list[tuple[int, int]]:
    matches: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start = sentence.find(text, cursor)
        if start == -1:
            break
        matches.append((start, start + len(text)))
        cursor = start + 1
    return matches


def validate_non_overlapping_spans(spans: list[Span]) -> list[str]:
    errors: list[str] = []
    ordered = sorted(spans, key=lambda item: (item.start, item.end))
    for previous, current in zip(ordered, ordered[1:]):
        if current.start < previous.end:
            errors.append(
                f"Overlapping spans detected: ({previous.start}, {previous.end}, {previous.label}) and "
                f"({current.start}, {current.end}, {current.label})."
            )
    return errors
