# Crosswalk Detection — Known Issues

The Grounding DINO crosswalk stage (`stages/crosswalk.py`) currently produces
mostly garbage boxes: tight rectangles around vehicles, road signs, and other
non-crosswalk content. This document records what's wrong and why, so the fix
can be designed before any model re-run.

## Symptom

In the dashboard's `CrosswalkCountQuery` results, "crosswalk" boxes are
small and tightly enclose vehicles rather than the painted road markings.
Counts per frame are also implausibly low for clearly-marked crossings.

## Empirical evidence

Sampling the first 50 segments' `crosswalks.json` (289 detections):

```
271  '##walk'        ← BERT subword piece
 12  'crosswalk'     ← intended label
  6  'cross'         ← BERT subword piece
```

All sampled confidences sit right at the threshold (0.30–0.35), i.e. the
detections barely pass and are almost certainly spurious sub-token matches.

## Root causes

### 1. Prompt is missing its terminating period

`stages/crosswalk.py:10`

```python
DEFAULT_PROMPT = "crosswalk"
```

Grounding DINO expects period-separated noun phrases (`"crosswalk."` or
`"a crosswalk. zebra crossing."`). The period is the phrase delimiter the
text-grounding head uses to know where one query phrase ends and the next
begins. Without it, the model has no grouped phrase to ground; it falls
back to grounding individual sub-word tokens of `crosswalk`, which the
BERT tokenizer splits into `["cross", "##walk"]`.

### 2. `text_threshold` is bound to `box_threshold`

`stages/crosswalk.py:86`

```python
all_results = self.processor.post_process_grounded_object_detection(
    gd_out, gd_in.input_ids,
    box_threshold=self.box_thr,
    text_threshold=self.box_thr,   # ← same arg used for both
    ...
)
```

`text_threshold` controls which text tokens are accepted as the box's
label. With both knobs glued together at `0.3`, a box passes whenever
*any single sub-word token* (e.g. `##walk`) clears 0.3 — there is no
requirement that the full noun phrase score above any threshold, and no
way to tune the two independently.

### 3. No label post-filtering

`stages/crosswalk.py:101-110`

```python
entries.append({
    "bbox": [...],
    "confidence": round(float(score), 3),
    "label": label,           # ← stored verbatim
})
```

Whatever string the post-processor returns is written to disk. There is
no check that the label is the intended noun phrase, no normalisation
of sub-word pieces, and no rejection of `##walk` / `cross` orphans.

### 4. Ingest hides the bug downstream

`curation/ingest.py:157`

```python
"""INSERT INTO detections
   (..., class_label, confidence, ...)
   VALUES (?, 'crosswalk', ?, ...)"""
```

The DB column `detections.class_label` is hard-coded to `'crosswalk'`
when ingesting `crosswalks.json`, regardless of the actual `label`
field. This means dashboard queries filtering on
`class_label = 'crosswalk'` happily return the spurious `##walk` boxes
as if they were real crosswalk detections — the upstream label problem
is invisible at query time.

## Why this produces "vehicle boxes"

Grounding DINO's text-image contrastive head learned associations
between subword tokens and image regions during pre-training. The
`##walk` sub-token co-occurs with urban street imagery (sidewalks,
walkers, vehicles, road furniture) often enough that it has weak
positive affinity to many of those regions. With:

- no phrase boundary (root cause 1) → grounding operates on sub-tokens
- a permissive threshold (root cause 2) → weak affinities pass
- no post-filter (root cause 3) → everything is written

the result is dense, low-confidence sub-token detections that look like
"random salient objects in urban scenes" — which empirically often turn
out to be vehicles.

## Recommended fixes (not yet applied)

Apply roughly in this order; (1) alone may already resolve most of the
problem.

1. **Add the period.** [APPLIED] Prompt changed to `"crosswalk."` in
   `stages/crosswalk.py:10`. The argparse help string was also updated
   to flag the period requirement. The dataset still needs a re-run to
   pick this up; existing `crosswalks.json` files are stale.

2. **Decouple the two thresholds.** Add a separate
   `--crosswalk-text-threshold` flag (default e.g. `0.25`) and pass it
   as `text_threshold=…` instead of reusing `box_thr`. With a real noun
   phrase from fix (1), text_threshold can stay reasonably low and still
   produce sensible labels.

3. **Filter labels post-hoc.** [APPLIED] `stages/crosswalk.py` now
   derives `self.canon_labels` from the prompt (period-split, lower-cased)
   and skips any detection whose label is empty, starts with `##`, or
   isn't one of those canonical phrases. Verified on the 2-segment
   sample: 41/41 surviving detections are labelled `crosswalk`, zero
   subword leakage.

4. **Stop hard-coding `class_label`.** In `curation/ingest.py:157`,
   either honour the JSON label or assert it matches the expected
   phrase before inserting. This way future label-quality regressions
   show up immediately in DB queries instead of being silently masked.

5. **Validate before re-ingest.** After re-running the stage on a
   handful of segments, eyeball the dashboard's `CrosswalkCountQuery`
   results to confirm boxes land on painted road markings rather than
   vehicles. The label distribution check used above (count labels
   across sampled `crosswalks.json` files) is a quick automated smoke
   test.

## Cost of a re-run

Grounding DINO base @ FP16, batch 8, ~30 fps source. The full dataset
re-run is the dominant cost; partial re-runs over a few segments for
validation are cheap (minutes per segment on one GPU).
