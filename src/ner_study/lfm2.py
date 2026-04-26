"""Fine-tuning and evaluation for LFM2-350M as a generative extractor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ner_study.config import Lfm2Config
from ner_study.instrumentation import current_rss_mb, timed_block
from ner_study.io_utils import ensure_dir, write_json, write_text
from ner_study.metrics import compute_span_metrics
from ner_study.modeling_utils import batched, get_torch_device, get_torch_dtype, move_batch_to_device
from ner_study.prompts import build_lfm2_chat_prompt, build_lfm2_chat_training_text
from ner_study.schemas import AggregateMetrics, SampleMetrics, SentenceExample
from ner_study.span_utils import parse_json_text_label_spans, spans_to_text_label_json


@dataclass(slots=True)
class GenerativeTrainExample:
    example: SentenceExample
    prompt: str
    target: str
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


class GenerativeDataset(Dataset):
    def __init__(self, items: list[GenerativeTrainExample]) -> None:
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


def build_train_examples(tokenizer, examples: list[SentenceExample], config: Lfm2Config) -> list[GenerativeTrainExample]:
    built: list[GenerativeTrainExample] = []
    for example in examples:
        target = spans_to_text_label_json(example.spans)
        prompt = build_lfm2_chat_prompt(tokenizer, example.sentence)
        full_text = build_lfm2_chat_training_text(tokenizer, example.sentence, target)
        prompt_ids = tokenizer(prompt, truncation=True, max_length=config.max_source_length, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(
            full_text,
            truncation=True,
            max_length=config.max_source_length + config.max_target_length,
            add_special_tokens=False,
        )["input_ids"]
        target_ids = full_ids[len(prompt_ids) :]
        input_ids = prompt_ids + target_ids
        attention_mask = [1] * len(input_ids)
        labels = ([-100] * len(prompt_ids)) + target_ids
        built.append(
            GenerativeTrainExample(
                example=example,
                prompt=prompt,
                target=target,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        )
    return built


def collate_generative_batch(batch: list[dict], pad_token_id: int) -> dict:
    max_length = max(len(item["input_ids"]) for item in batch)
    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []
    for item in batch:
        pad_length = max_length - len(item["input_ids"])
        padded_input_ids.append(item["input_ids"] + ([pad_token_id] * pad_length))
        padded_attention_mask.append(item["attention_mask"] + ([0] * pad_length))
        padded_labels.append(item["labels"] + ([-100] * pad_length))
    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
    }


def resolve_lora_target_modules(model, requested: tuple[str, ...]) -> list[str]:
    available_module_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    resolved = [name for name in requested if name in available_module_names]
    if resolved:
        return resolved
    fallback = [name for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj") if name in available_module_names]
    if not fallback:
        raise ValueError("Could not determine LoRA target modules for the current model.")
    return fallback


def train_lfm2(
    train_examples: list[SentenceExample],
    validation_examples: list[SentenceExample],
    test_examples: list[SentenceExample],
    config: Lfm2Config,
    output_dir: Path,
) -> tuple[AggregateMetrics, Path]:
    device = get_torch_device()
    dtype = get_torch_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=dtype)
    lora_modules = resolve_lora_target_modules(base_model, config.target_modules)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=lora_modules,
    )
    model = get_peft_model(base_model, peft_config).to(device)

    train_items = build_train_examples(tokenizer, train_examples, config)
    validation_items = build_train_examples(tokenizer, validation_examples, config)
    train_loader = DataLoader(
        GenerativeDataset(train_items),
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_generative_batch(batch, tokenizer.pad_token_id),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    best_state_path = output_dir / "best"
    ensure_dir(best_state_path)
    best_f1 = -1.0
    patience = 0

    with timed_block() as train_timer:
        for _epoch_index in range(int(config.num_train_epochs)):
            model.train()
            for step_index, batch in enumerate(train_loader, start=1):
                batch = move_batch_to_device(batch, device)
                loss = model(**batch).loss / config.gradient_accumulation_steps
                loss.backward()
                if step_index % config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            if len(train_loader) % config.gradient_accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            validation_predictions, _validation_metrics = generate_lfm2_predictions(
                model=model,
                tokenizer=tokenizer,
                examples=[item.example for item in validation_items],
                max_new_tokens=config.max_target_length,
                batch_size=config.eval_batch_size,
            )
            validation_metrics = compute_span_metrics(
                system_name="lfm2_validation",
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
                if patience >= 2:
                    break

    reloaded_tokenizer = AutoTokenizer.from_pretrained(best_state_path)
    if reloaded_tokenizer.pad_token is None:
        reloaded_tokenizer.pad_token = reloaded_tokenizer.eos_token
    reloaded_base_model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=dtype)
    inference_model = PeftModel.from_pretrained(reloaded_base_model, best_state_path).to(device)

    pred_spans, sample_metrics = generate_lfm2_predictions(
        model=inference_model,
        tokenizer=reloaded_tokenizer,
        examples=test_examples,
        max_new_tokens=config.max_target_length,
        batch_size=config.eval_batch_size,
        raw_output_dir=output_dir / "raw_outputs",
    )
    metrics = compute_span_metrics(
        system_name="lfm2_350m_sft",
        gold_batches=[example.spans for example in test_examples],
        pred_batches=pred_spans,
        sample_metrics=sample_metrics,
        train_runtime_seconds=train_timer.duration_seconds,
    )
    predictions_path = output_dir / "lfm2_predictions.json"
    write_json(
        predictions_path,
        [
            {
                "sample_id": example.id,
                "sentence": example.sentence,
                "gold_spans": [span.model_dump() for span in example.spans],
                "predicted_spans": [span.model_dump() for span in spans],
            }
            for example, spans in zip(test_examples, pred_spans, strict=True)
        ],
    )
    return metrics, predictions_path


@torch.no_grad()
def generate_lfm2_predictions(
    model,
    tokenizer,
    examples: list[SentenceExample],
    max_new_tokens: int,
    batch_size: int = 1,
    raw_output_dir: Path | None = None,
) -> tuple[list[list], list[SampleMetrics]]:
    model.eval()
    device = next(model.parameters()).device
    predictions: list[list] = []
    sample_metrics: list[SampleMetrics] = []

    if raw_output_dir is not None:
        ensure_dir(raw_output_dir)

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    for batch_examples in batched(examples, batch_size):
        prompts = [build_lfm2_chat_prompt(tokenizer, example.sentence) for example in batch_examples]
        model_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        prompt_token_counts = [int(mask.sum().item()) for mask in model_inputs["attention_mask"]]
        with timed_block() as timer:
            generated = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        input_width = int(model_inputs["input_ids"].shape[-1])

        for local_index, example in enumerate(batch_examples):
            generated_tokens = generated[local_index][input_width:]
            raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            spans, parse_errors = parse_json_text_label_spans(raw_text, example.sentence)
            predictions.append(spans)

            raw_output_path = None
            if raw_output_dir is not None:
                raw_output_path = str(raw_output_dir / f"{example.id}.txt")
                write_text(
                    raw_output_path,
                    raw_text + ("\n\nPARSE_ERRORS:\n" + "\n".join(parse_errors) if parse_errors else ""),
                )
            completion_tokens = int(generated_tokens.shape[-1])
            prompt_tokens = prompt_token_counts[local_index]
            sample_metrics.append(
                SampleMetrics(
                    sample_id=example.id,
                    latency_seconds=timer.duration_seconds / len(batch_examples),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    memory_rss_mb=current_rss_mb(),
                    raw_output_path=raw_output_path,
                )
            )

    tokenizer.padding_side = original_padding_side
    return predictions, sample_metrics
