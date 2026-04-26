# AI Benchmark

Small benchmark to compare three ways of doing NER on `CoNLL-2003`:

- `bert-base-cased` as a classic predictive NER baseline
- `LiquidAI/LFM2-350M` fine-tuned to return JSON entities
- `Qwen` served locally with `ollama` and prompted for extraction

All three are evaluated with the same span-based scorer on the same labels:
`PER`, `ORG`, `LOC`, `MISC`.

## What is in this repo

- `src/ner_study/`: benchmark code
- `python -m ner_study.cli benchmark`: full run
- `python -m ner_study.cli smoke`: tiny smoke test

The benchmark:

- loads `tomaarsen/conll2003`
- reconstructs each sentence from tokens
- evaluates every system as character-level spans
- writes metrics, predictions, timings, token usage, and cost estimates

## Install

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you also want to run tests:

```bash
pip install -e '.[dev]'
pytest
```

## Model setup

`BERT` and `LFM2` are loaded from Hugging Face by the Python runners.

`Qwen` is served through `ollama`.

Example:

```bash
ollama pull qwen3.6:35b-a3b-mxfp8
```

For the runtime-only comparison of a small Liquid model in `ollama`:

```bash
ollama pull LiquidAI/lfm2.5-350m
```

## Run

Smoke test:

```bash
NER_STUDY_TORCH_DEVICE=cpu python -m ner_study.cli smoke \
  --output-dir artifacts/smoke \
  --qwen-model qwen3.6:35b-a3b-mxfp8 \
  --smoke-samples 2 \
  --skip-lfm2-train
```

Full benchmark:

```bash
NER_STUDY_TORCH_DEVICE=cpu python -m ner_study.cli benchmark \
  --output-dir artifacts/full \
  --qwen-model qwen3.6:35b-a3b-mxfp8
```

Notes:

- `NER_STUDY_TORCH_DEVICE=cpu` is the safest default on local Mac setups.
- If you already have a larger Qwen model in `ollama`, pass it with `--qwen-model`.
- `--skip-lfm2-train` skips the LFM2 fine-tune stage.

## Outputs

After a run, the main files are:

- `artifacts/<run>/comparison.md`: summary table
- `artifacts/<run>/baseline/metrics.json`
- `artifacts/<run>/lfm2/metrics.json`
- `artifacts/<run>/qwen/metrics.json`

Each system also writes normalized predictions and raw outputs.

## Results


| System | F1 | Recall | Mean latency | Cost |
| --- | ---: | ---: | ---: | ---: |
| `baseline_bert` | `0.9049` | `0.9108` | `0.0455s` | `X` |
| `lfm2_350m_sft` | `0.8569` | `0.8415` | `0.3187s` | `21X` |
| `qwen_ollama_qwen3.6_35b-a3b-mxfp8` | `0.7675` | `0.6833` | `1.2929s` | `296X` |

Notes:

- `LFM2` latency and cost are shown through the `ollama` serving proxy used in this study, with a `g4dn.xlarge` serving profile.
- `Qwen3.6` cost is estimated with a `g6e.xlarge` serving profile.
