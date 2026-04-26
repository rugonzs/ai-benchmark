"""Dataset loading and transformation for CoNLL-2003."""

from __future__ import annotations

from collections.abc import Iterable

from datasets import Dataset, DatasetDict, load_dataset

from ner_study.config import DataConfig
from ner_study.schemas import SentenceExample
from ner_study.span_utils import iob_tags_to_spans
from ner_study.text_utils import detokenize_with_offsets


def load_conll2003(config: DataConfig) -> DatasetDict:
    dataset = load_dataset(config.dataset_name)
    return DatasetDict(
        {
            "train": maybe_take(dataset["train"], config.max_train_samples),
            "validation": maybe_take(dataset["validation"], config.max_validation_samples),
            "test": maybe_take(dataset["test"], config.max_test_samples),
        }
    )


def maybe_take(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None or limit >= len(dataset):
        return dataset
    return dataset.select(range(limit))


def convert_split_to_examples(split_name: str, split_dataset: Dataset) -> list[SentenceExample]:
    examples: list[SentenceExample] = []
    features = split_dataset.features["ner_tags"].feature
    for index, row in enumerate(split_dataset):
        tokens = [str(token) for token in row["tokens"]]
        ner_tags = [features.int2str(int(tag)) for tag in row["ner_tags"]]
        sentence, offsets = detokenize_with_offsets(tokens)
        spans = iob_tags_to_spans(tokens, ner_tags, offsets)
        examples.append(
            SentenceExample(
                split=split_name,
                id=f"{split_name}-{index:05d}",
                tokens=tokens,
                ner_tags=ner_tags,
                sentence=sentence,
                spans=spans,
            )
        )
    return examples


def load_sentence_examples(config: DataConfig) -> dict[str, list[SentenceExample]]:
    dataset = load_conll2003(config)
    return {
        split_name: convert_split_to_examples(split_name, split_dataset)
        for split_name, split_dataset in dataset.items()
    }


def examples_to_dicts(examples: Iterable[SentenceExample]) -> list[dict]:
    return [example.model_dump(mode="json") for example in examples]
