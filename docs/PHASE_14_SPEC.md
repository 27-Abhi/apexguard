# Phase 14 Spec — Speech ML & ASR Specialization (Hugging Face / LoRA / PEFT)

## Overview
Phase 14 delivers ML model development evidence for speech roles by implementing automated ASR evaluation (WER/CER metrics) and Parameter-Efficient Fine-Tuning (LoRA/PEFT) on open-source audio datasets.

## Module Pipeline

```text
Audio Dataset (LibriSpeech/CommonVoice) -> Preprocessing -> Whisper/Conformer ASR -> Evaluate (WER / CER) -> Error Analysis
                                                                    │
                                                                    ▼
                                                  PEFT / LoRA Adapter Fine-Tuning
```

## Evaluated Speech Metrics

1. **Word Error Rate (WER):** $\text{WER} = \frac{S + D + I}{N}$ where $S$ = Substitutions, $D$ = Deletions, $I$ = Insertions, $N$ = Total words.
2. **Character Error Rate (CER):** Character-level error breakdown for domain-specific terminology.

## Deliverables

- `tests/eval_asr.py` script calculating WER/CER across test audio splits.
- PEFT/LoRA adapter configuration script fine-tuning Whisper on target domain audio.
