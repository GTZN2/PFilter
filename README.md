
# PFilter: Prototypical Self-Evaluation Filter for LLM-based NER

**Overview.** PFilter is a lightweight, training-free plugin that reduces **type misidentifications** made by LLMs during Named Entity Recognition (NER). It asks the LLM to **self-evaluate** the relevance of each predicted entity to a target type, learns **rating prototypes** from a small calibration set, and then filters likely mis-typed entities at inference&#x20;

---

## How it works

PFilter converts an LLM’s **self-evaluation ratings** into **prototypes** and applies a two-level decision rule at inference.

* **Step 1 — Self-evaluate.** Given a sentence (or a list of candidate entities) and a target type (PER/LOC/ORG), the LLM outputs entities **plus a 1–10 relevance rating** indicating how well each entity matches the target type. This is done in one turn (PFilter-S) or two turns (PFilter-D), optionally also asking the model to list **non-target** entities and rate their relevance to the target.&#x20;
* **Step 2 — Prototype induction.** From a **small calibration set**, PFilter summarizes the empirical rating distributions into **class prototypes** (for “correct” vs “incorrect”). Ratings for correct vs. incorrect entities show **clear clustering** and **directional consistency**, which the filter exploits.&#x20;
* **Step 3 — Filtering at inference.** For new sentences, PFilter elicits ratings in the same protocol and removes entities that fall in **Level-1** strict rejection regions or, in **Level-2**, are more likely under the “incorrect” prototype distribution (via a likelihood-ratio/Mahalanobis test). (See the formal region definitions for both single- and dual-turn settings.)&#x20;

### Two variants

* **PFilter-S (single turn):** Lower interaction overhead; folds target and non-target rating into one turn, then filters.&#x20;
* **PFilter-D (dual turn):** Slightly more interactive but reduces per-turn context load (helpful under tight context or latency constraints).&#x20;

---

## Repository structure

```
PFilter/
├─ dataset/              # Data for eval
├─ offline/              # Offline filtering utility for models deployed on Ollama 
├─ online/               # Online filtering utilities for GPT-4.1/GPT-3.5 Turbo
├─ PrototypeGeneration.py# Build rating prototypes (PFilter-S/D)
├─ nerEva.py             # Run NER eval
└─ README.md             #  
```

---
