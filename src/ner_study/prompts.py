"""Prompt templates for generative NER models."""

from __future__ import annotations


SYSTEM_PROMPT = """You are a strict CoNLL-2003 named entity extraction system.
Base your answer only on the target sentence.
Examples in the prompt are demonstrations only. Never copy entities from the examples.
If the target sentence has no explicit named entity, return [].
Return JSON only. Do not explain your answer.
"""

LFM2_SYSTEM_PROMPT = "Extract CoNLL-2003 named entities. Return only valid JSON."


def build_extraction_prompt(sentence: str) -> str:
    return f"""Extract CoNLL-2003 named entities from the sentence below.

You must detect only these labels:
- PER: specific people, full names, surnames, player names, coach names, politician names, etc.
- ORG: named organizations such as companies, agencies, clubs, institutions, newspapers, political parties, competitions when tagged as organizations, and national teams only when they function as organizations.
- LOC: named places such as countries, cities, regions, provinces, states, rivers, seas, mountains, and other geographical locations.
- MISC: named entities that are not PER, ORG, or LOC, mainly nationalities, demonyms, named events, and other clearly named proper expressions. MISC is still for named entities, not for generic content words.

Decision rules:
- Use only the target sentence. Do not use or copy entities from any instruction text.
- Every extracted item must be an exact contiguous substring of the target sentence.
- Extract only explicit named entities, not inferred entities.
- If a candidate is a generic word or phrase, reject it.
- If the target sentence is only punctuation, symbols, formatting markers, or other non-entity fragments, return [].
- If the sentence is a table row, standings row, score summary, market/statistics line, quote fragment, or broken sentence fragment, return [] unless it still contains an explicit named entity.
- In sports results and lineups, team or club names are usually ORG, while player names are PER. Country names remain LOC.
- In roster or lineup formats, person names are still PER even when they appear after numbers or punctuation.
- If an entity contains punctuation, apostrophes, or hyphens, copy the full entity exactly as written.

Do not extract:
- dates, years, times, scores, percentages, currencies, quantities
- common nouns, verbs, adjectives, sports, occupations, roles, positions
- generic phrases like "the government", "the team", "the match", "the company"
- one-word generic nouns such as "terrorists", "market", "buyers", "music", or "software" when they are not part of a named entity
- headline category words like "SOCCER" or "RUGBY" unless they are clearly part of a named entity
- table headers, standings labels, market data fields, score summaries, fragments of quotes, or broken sentence fragments unless they contain an explicit named entity
- text that is not explicitly present in the sentence

Output format:
- Return exactly one JSON array.
- Each item must be an object with keys: "text" and "label".
- Allowed labels are only: "PER", "ORG", "LOC", "MISC".
- Do not return offsets.
- Do not return markdown.
- Do not return explanations.
- Copy each entity text exactly as it appears in the target sentence, character for character.
- Do not repeat the same mention twice.
- Do not return overlapping entities.
- Never output the label names "PER", "ORG", "LOC", or "MISC" as entity text.
- If uncertain, prefer returning fewer entities over inventing generic ones.
- If there are no entities, return [].

Target sentence:
<sentence>
{sentence}
</sentence>
"""


def build_chat_extraction_prompt(tokenizer, sentence: str) -> str:
    """Build a model-native chat prompt when the tokenizer exposes a chat template."""

    user_prompt = build_extraction_prompt(sentence)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    return SYSTEM_PROMPT + "\n" + user_prompt


def build_lfm2_user_prompt(sentence: str) -> str:
    return (
        "Task: extract only CoNLL-2003 named entities from the sentence.\n"
        'Allowed labels: "PER", "ORG", "LOC", "MISC".\n'
        "Only extract explicit named entities written in the sentence.\n"
        "Do not extract dates, scores, numbers, roles, occupations, verbs, or generic nouns.\n"
        'Return exactly a JSON array of objects with keys "text" and "label".\n'
        'Valid example: [{"text":"EU","label":"ORG"},{"text":"German","label":"MISC"}]\n'
        'Valid empty output: []\n'
        "Do not use markdown.\n"
        "Do not add comments.\n"
        "Do not invent entities.\n"
        f"Sentence: {sentence}\n"
        "JSON:"
    )


def build_lfm2_chat_prompt(tokenizer, sentence: str) -> str:
    user_prompt = build_lfm2_user_prompt(sentence)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": LFM2_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    return LFM2_SYSTEM_PROMPT + "\n" + user_prompt + "\n"


def build_lfm2_chat_training_text(tokenizer, sentence: str, target_json: str) -> str:
    user_prompt = build_lfm2_user_prompt(sentence)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": LFM2_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": target_json},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
    return LFM2_SYSTEM_PROMPT + "\n" + user_prompt + "\n" + target_json
