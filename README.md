# Frankenstein QLoRA: Mistral 7B finetuned on Mary Shelley's 1818 novel

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/trustkrem/frankenstein-qlora-mistral7b/blob/main/frankenstein_qlora_mistral7b.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Parameter-efficient finetuning (QLoRA) of **Mistral 7B** on the text of *Frankenstein; or, The Modern Prometheus* (1818), so the model writes original Frankenstein-style fanfiction. Built as a portfolio project for Codecademy's *Finetuning Transformer Models* course.

**Result:** test-set perplexity dropped from **9.09 → 6.83 (−25%)**, and generations shifted from modern blog voice to 19th-century gothic prose. Training updated only **0.19%** of the model's parameters and took **~16 minutes on a free Colab T4 GPU**.

---

## Results

### Perplexity (97 held-out chunks, never seen in training)

| Model | Perplexity ↓ |
|---|---|
| Mistral 7B (base, 4-bit) | 9.09 |
| Mistral 7B + Frankenstein LoRA | **6.83** |

### Same prompt, before and after

Prompt: `"I'm afraid I've created a "` (greedy decoding, 100 new tokens)

**Base model:**
```
I'm afraid I've created a monster.

I've been working on a new project for the past few months, and I'm finally
ready to share it with you.

It's called the "Mindful Money Masterclass" and it's a 6-week course that will
teach you how to make money without sacrificing your values.
```

**Finetuned model:**
```
I'm afraid I've created a monster in putting life and sensation into dead
matter. The one that your sister saved from the gallows may have been justified
in desiring even a premature death; but I am unworthy of your trust. Let me not
be confounded with the wretch to whom I gave existence. I shall escape him, and
find a place for repentance. Adieu!

"Your affectionate friend and
```

The base model recognizes the "monster" association but immediately drifts into the newsletter register that dominates its web pretraining data. The finetuned model holds the novel's first-person gothic voice, its vocabulary ("wretch", "gave existence", "Adieu"), its plot elements, and even its epistolary (letter-writing) structure.

---

## Method

### Data
- 481 text chunks from the Project Gutenberg edition of *Frankenstein* (`frankenstein_chunks.csv`)
- EDA: no nulls, no duplicates; a few chunks contain Gutenberg header/license boilerplate
- 80/20 split → **384 train / 97 test** chunks, converted to Hugging Face `Dataset`s and tokenized with the Mistral tokenizer

### 4-bit quantization (bitsandbytes)
| Setting | Value |
|---|---|
| `load_in_4bit` | `True` |
| `bnb_4bit_quant_type` | `"nf4"` (NormalFloat, suited to normally distributed weights) |
| `bnb_4bit_use_double_quant` | `True` (also quantizes the quantization constants) |
| `bnb_4bit_compute_dtype` | `torch.bfloat16` |

### LoRA (PEFT)
| Setting | Value |
|---|---|
| Rank `r` | 32 |
| `lora_alpha` | 64 |
| `lora_dropout` | 0.05 |
| Target modules | `q_proj`, `v_proj` (PEFT default for Mistral) |
| Task type | `CAUSAL_LM` |

**Trainable parameters: 13,631,488 of 7,255,363,584 (0.19%).** This can be verified by hand: each adapter adds `r × (in + out)` parameters.

```
q_proj: 32 × (4096 + 4096) = 262,144
v_proj: 32 × (4096 + 1024) = 163,840   (grouped-query attention: K/V project to 1024)
per layer: 425,984 × 32 decoder layers = 13,631,488 ✓
```

### Training
| Setting | Value |
|---|---|
| Batch size | 2 |
| Epochs | 2 (384 steps) |
| Learning rate | 2e-5, 2 warmup steps |
| Optimizer | `paged_adamw_8bit` |
| Precision | fp16 mixed precision |
| Hardware / time | Colab T4 (16 GB) / ~16.5 min |

Training loss fell from ~1.89 (mean of first 20 steps) to ~1.66 (mean of last 20), with a mean of 1.81 over the run. There was no sharp drop at the epoch boundary, a sign the model wasn't simply memorizing the training chunks on its second pass.

### Evaluation
- **Quantitative:** mean per-chunk perplexity (`exp(cross-entropy loss)`) on the held-out test set, with the model in `eval()` mode for both measurements
- **Qualitative:** greedy-decoded completions of the same prompt before and after finetuning

---

## Limitations

- **In-domain test set.** Test chunks come from the same novel as the training chunks, so the perplexity gain shows strong adaptation to *this book*, not general creative-writing ability.
- **Likely pretraining exposure.** *Frankenstein* is public domain and was probably in Mistral's pretraining data, which helps explain the already low base perplexity.
- **Formatting artifacts.** Gutenberg's hard-wrapped lines (a newline every ~70 characters) were kept in the data, so the model reproduces them.
- **Plot drift.** The model remixes story elements freely (e.g. who saved whom from the gallows), which suits fanfiction but not summarization.

## Next experiments

- Remove Gutenberg boilerplate and join hard-wrapped lines, then compare perplexity and output formatting
- Target more modules (`k_proj`, `o_proj`, MLP projections) and compare quality against cost
- Sweep rank (`r=8` vs `r=64`), epochs and learning rate, watching test perplexity for the onset of overfitting
- Sampling (`do_sample=True`, `temperature≈0.8`) for more varied, less repetitive fanfiction

---

## Run it yourself

1. Click **Open in Colab** above and select **Runtime → Change runtime type → T4 GPU**
2. Upload `frankenstein_chunks.csv` from this repo via Colab's file panel
3. (Recommended) Add a Hugging Face read token as a Colab secret named `HF_TOKEN`
4. Run all cells (expect ~5 min model download + ~16 min training)

## Use the trained adapter

The LoRA adapter (~55 MB) is hosted on Hugging Face: [trustkrem/frankenstein-qlora-mistral7b](https://huggingface.co/trustkrem/frankenstein-qlora-mistral7b)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_id = "mistralai/Mistral-7B-v0.1"
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base = AutoModelForCausalLM.from_pretrained(base_id, quantization_config=quant_config, device_map="auto")
model = PeftModel.from_pretrained(base, "trustkrem/frankenstein-qlora-mistral7b")
tokenizer = AutoTokenizer.from_pretrained(base_id)

inputs = tokenizer("I'm afraid I've created a ", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Tech stack

PyTorch · Hugging Face Transformers · PEFT · bitsandbytes · Datasets · Accelerate · pandas · Google Colab

## Acknowledgments

- Project brief and starter notebook: [Codecademy](https://www.codecademy.com/), *Finetuning Transformer Models*
- Source text: Mary Shelley, *Frankenstein* (1818), via [Project Gutenberg](https://www.gutenberg.org/ebooks/84) (public domain)
- Base model: [Mistral 7B v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) by Mistral AI (Apache 2.0)
