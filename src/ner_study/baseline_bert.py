"""Supervised non-generative baseline using bert-base-cased token classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification

from ner_study.config import BaselineConfig
from ner_study.constants import IOB_LABELS
from ner_study.instrumentation import current_rss_mb, timed_block
from ner_study.io_utils import ensure_dir, write_json
from ner_study.metrics import compute_span_metrics
from ner_study.modeling_utils import batched, get_torch_device, move_batch_to_device
from ner_study.schemas import AggregateMetrics, SampleMetrics, SentenceExample
from ner_study.span_utils import iob_tags_to_spans
from ner_study.text_utils import detokenize_with_offsets


LABEL2ID = {label: index for index, label in enumerate(IOB_LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}


@dataclass(slots=True)
class EncodedTokenExample:
    example: SentenceExample
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    word_ids: list[int | None]


class TokenDataset(Dataset):
    def __init__(self, items: list[EncodedTokenExample]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        return {
            "input_ids": item.input_ids,
            "attention_mask": item.attention_mask,
            "labels": item.labels,
        }


def encode_examples(tokenizer, examples: list[SentenceExample], max_length: int) -> list[EncodedTokenExample]:
    encoded_items: list[EncodedTokenExample] = []
    for example in examples:
        encoded = tokenizer(
            example.tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        word_ids = encoded.word_ids()
        labels: list[int] = []
        seen_word_ids: set[int] = set()
        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
                continue
            if word_id in seen_word_ids:
                labels.append(-100)
                continue
            seen_word_ids.add(word_id)
            labels.append(LABEL2ID[example.ner_tags[word_id]])

        encoded_items.append(
            EncodedTokenExample(
                example=example,
                input_ids=list(encoded["input_ids"]),
                attention_mask=list(encoded["attention_mask"]),
                labels=labels,
                word_ids=word_ids,
            )
        )
    return encoded_items


def build_dataloader(items: list[EncodedTokenExample], tokenizer, batch_size: int, shuffle: bool) -> DataLoader:
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    return DataLoader(TokenDataset(items), batch_size=batch_size, shuffle=shuffle, collate_fn=collator)


def train_baseline(
    train_examples: list[SentenceExample],
    validation_examples: list[SentenceExample],
    test_examples: list[SentenceExample],
    config: BaselineConfig,
    output_dir: Path,
) -> tuple[AggregateMetrics, Path]:
    device = get_torch_device()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        config.model_name,
        num_labels=len(IOB_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    train_items = encode_examples(tokenizer, train_examples, config.max_length)
    validation_items = encode_examples(tokenizer, validation_examples, config.max_length)

    train_loader = build_dataloader(train_items, tokenizer, config.train_batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_f1 = -1.0
    best_state_path = output_dir / "best"
    ensure_dir(best_state_path)
    patience = 0

    with timed_block() as train_timer:
        for _epoch_index in range(int(config.num_train_epochs)):
            model.train()
            for batch in train_loader:
                batch = move_batch_to_device(batch, device)
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            validation_predictions, _ = predict_baseline(model, tokenizer, validation_items, config.eval_batch_size)
            validation_metrics = compute_span_metrics(
                system_name="baseline_validation",
                gold_batches=[item.example.spans for item in validation_items],
                pred_batches=validation_predictions,
            )
            if validation_metrics.micro_f1 > best_f1:
                best_f1 = validation_metrics.micro_f1
                patience = 0
                model.save_pretrained(best_state_path)
                tokenizer.save_pretrained(best_state_path)
            else:
                patience += 1
                if patience >= config.early_stopping_patience:
                    break

    best_model = AutoModelForTokenClassification.from_pretrained(best_state_path).to(device)
    best_tokenizer = AutoTokenizer.from_pretrained(best_state_path)
    test_items = encode_examples(best_tokenizer, test_examples, config.max_length)
    pred_spans, sample_metrics = predict_baseline(best_model, best_tokenizer, test_items, batch_size=1)
    metrics = compute_span_metrics(
        system_name="baseline_bert",
        gold_batches=[item.example.spans for item in test_items],
        pred_batches=pred_spans,
        sample_metrics=sample_metrics,
        train_runtime_seconds=train_timer.duration_seconds,
    )

    predictions_path = output_dir / "baseline_predictions.json"
    write_json(
        predictions_path,
        [
            {
                "sample_id": item.example.id,
                "sentence": item.example.sentence,
                "gold_spans": [span.model_dump() for span in item.example.spans],
                "predicted_spans": [span.model_dump() for span in spans],
            }
            for item, spans in zip(test_items, pred_spans, strict=True)
        ],
    )
    return metrics, predictions_path


@torch.no_grad()
def predict_baseline(
    model,
    tokenizer,
    items: list[EncodedTokenExample],
    batch_size: int,
) -> tuple[list[list], list[SampleMetrics]]:
    device = next(model.parameters()).device
    model.eval()
    predictions: list[list] = []
    sample_metrics: list[SampleMetrics] = []
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    for batch_items in batched(items, batch_size):
        encoded_batch = collator(
            [
                {
                    "input_ids": item.input_ids,
                    "attention_mask": item.attention_mask,
                    "labels": item.labels,
                }
                for item in batch_items
            ]
        )
        with timed_block() as timer:
            outputs = model(**move_batch_to_device(encoded_batch, device))
        logits = outputs.logits.detach().cpu()

        for local_index, item in enumerate(batch_items):
            predicted_tag_ids = logits[local_index].argmax(dim=-1).tolist()
            predicted_tags = decode_predicted_tags(item.word_ids, predicted_tag_ids, len(item.example.tokens))
            _sentence, offsets = detokenize_with_offsets(item.example.tokens)
            predicted_spans = iob_tags_to_spans(item.example.tokens, predicted_tags, offsets)
            predictions.append(predicted_spans)

            sample_metrics.append(
                SampleMetrics(
                    sample_id=item.example.id,
                    latency_seconds=timer.duration_seconds / len(batch_items),
                    prompt_tokens=len(item.input_ids),
                    completion_tokens=0,
                    total_tokens=len(item.input_ids),
                    memory_rss_mb=current_rss_mb(),
                )
            )

    return predictions, sample_metrics


def decode_predicted_tags(word_ids: list[int | None], predicted_tag_ids: list[int], token_count: int) -> list[str]:
    tag_by_word: dict[int, str] = {}
    for word_id, tag_id in zip(word_ids, predicted_tag_ids):
        if word_id is None or word_id in tag_by_word:
            continue
        tag_by_word[word_id] = ID2LABEL[int(tag_id)]
    return [tag_by_word.get(index, "O") for index in range(token_count)]
