# Gold Annotation Guide — NER vs. LLM Parsing

## Scope and label taxonomy

**Core evaluation**: `PERSON`, `ORG`, `LOC`. English spaCy's `GPE` and `LOC` are both mapped to `LOC`. This deliberately collapses the geopolitical / physical-location distinction. `PER` also maps to `PERSON`.

**Auxiliary categories**: `DATE`, `PRODUCT`, `JOB_TITLE`. `en_core_web_sm` already has `DATE` and `PRODUCT` labels, **but not `JOB_TITLE`**. Do not describe `PRODUCT` as unsupported by English spaCy. In the *category flexibility* experiment, measure LLM `JOB_TITLE` extraction and separately document that the existing NER output head has no `JOB_TITLE` label. A fine-tuned or rule-augmented NER system could acquire one.

## Annotation principles

1. Tag only explicit text spans, never inferred attributes or paraphrases. If a location isn't present, use no `LOC` span.
2. Entity = complete proper-name mention, excluding `Dr.` honorifics and adjacent punctuation. `Sarah Johnson` is one `PERSON`. An abbreviated later mention `Sarah` is also a `PERSON` when it refers to the individual.
3. `Apple` named as a company is `ORG`; `apple` as fruit is outside. Annotate context-dependent names consistently.
4. `Jordan` (country) is `LOC`; `Jordan Blake` (person) is `PERSON`. Organization names containing city names are tagged as a complete `ORG` (e.g. `Washington University`).
5. In `Google'a` / `Londra'da`, **only the name stem** is in the span. Case suffix after apostrophe is out of span. The agreed policy must be applied to both gold and prediction evaluation.
6. Date text includes the complete explicitly stated date phrase (e.g. `March 12, 2025`); `2025` alone is `DATE` if it refers to a year.
7. `JOB_TITLE` marks job/role phrases when used as titles or occupations (e.g. `data scientist`), not personal names. `PRODUCT` marks explicit named products, not generic mentions of devices.
8. `start` is inclusive; `end` is exclusive in Python character offsets. Every gold span must satisfy `text[start:end] == entity['text']`. Repeated occurrences use distinct offsets.
9. `entities` contains ONLY `PERSON, ORG, LOC`; `extended_entities` contains `DATE, JOB_TITLE, PRODUCT`. Both together represent the complete annotated truth.
10. All paragraphs are invented educational scenarios, **not real-world claims about the companies or people named**. Manually review the gold labels before using them in slides or a published report.

## Scoring policy

- Main accuracy: exact character-span + normalized entity type, *micro*-aggregated TP/FP/FN. A partial name is not exact match.
- LLM answers contain `{ "entities": [{"text": "...", "label": "PERSON"}] }`. Surface strings are anchored back to the original source via deterministic case-insensitive substring matching. Repeated mentions are assigned in source order; any unanchorable extra string counts as an **unmatched false positive**.
- If JSON is invalid or fails schema validation, it is scored as an empty entity list (false negatives) and independently counted as a format error. Save raw outputs for error investigation.
- Use the **same subset of gold categories** for the two models. Never calculate cross-language NER results by running the English model on Turkish text as if it were a suitable Turkish NER baseline.
- `JSON parsing success`, `schema compliance`, and `entity accuracy` are different metrics. Keep them separate.
- Report synthetic dataset limitation and model checkpoint/version, CPU/GPU details, prompt, seed, temperature and quantization configuration.

## Research-design notes

- `development.jsonl` (5 English and 2 Turkish passages) is exclusively for choosing prompts and verifying parsing code. Do not include these in final test F1.
- `english_test.jsonl` has 50 test paragraphs in five groups (10 each): `standard`, `ambiguity`, `multi_entity`, `extended`, `challenging`.
- `turkish_supplementary.jsonl` has 10 Turkish passages paired with EN001–EN010 for **supplementary bilingual / morphology** analysis, NOT a separate full-scale NER comparison.
- `challenge` group is held out from prompt development. It is not independent of `english_test.jsonl`; its F1 should be reported as a **subset analysis**, not an independent external validation.
- Synthetic labels and model outputs are not research results until the models have actually been run and the annotations checked.
