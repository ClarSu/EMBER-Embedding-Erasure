# Reproduce optimal unlearning configs

This directory holds **pinned erasure configs** for rerunning the best hyperparameters found during grid search — one per `(method, model, concept)` triplet.

The source of truth is [`optimal_unlearning_hyperparams.yaml`](optimal_unlearning_hyperparams.yaml). Generated run configs live under:

```
<method>/<model>/<concept>.yaml
```

For example: `snmf/gemma/ancient_rome.yaml`.

## Goal

These configs are meant to **reproduce a single optimal unlearning run** and save the resulting checkpoint, not to repeat a full grid search. Each file therefore sets `topk: 1` and collapses method grids to one hyperparameter cell.

Non-grid settings (batch sizes, eval thresholds, etc.) are inherited from the corresponding base config in `configs/` (e.g. `crisp_llama.yaml`).

## Generate configs

```bash
python scripts/generate_reproduce_configs.py
```

Use `--input` to point at another metadata file (e.g. `optimal_unlearning_hyperparams_more_concepts.yaml`).

## Run a config

```bash
python -m ember.run_erasure \
    --config configs/reproduce_optimal_configs/snmf/gemma/ancient_rome.yaml \
    --concepts "Ancient Rome" --train-eval mc \
    --features-source local
```

Use `--train-eval open` for PISCES configs.

## Method-specific pinning

The standard erasure pipelines sweep some choices via hard-coded grids. For reproduction we added config fields to pin the winning combination directly.

### CRISP — `layer_ranges`

The default CRISP grid searches over a fixed set of LoRA layer spans. Reproduce configs set `layer_ranges` to the single winning triple `(layer_lo, layer_hi, layer_step)`, e.g. `[5, 19, 2]` on Llama.

### RMU — `update_settings`

The default RMU grid searches over preset layer targets. Each entry specifies:

- `setting_name` — label for logging/results (e.g. `S2_lid8_L678`)
- `layer_id` — layer whose activations drive the RMU loss
- `layer_ids` — layers whose weights receive gradient updates (typically MLP `down_proj`)

Example:

```yaml
update_settings:
  - setting_name: S2_lid8_L678
    layer_id: 8
    layer_ids: 6,7,8
```

### SNMF — `layer_ranges_in` / `layer_ranges_out`

The default SNMF grid searches over hard-coded MLP layer spans on both sides of the intervention:

- `layer_ranges_in` — layers where **up-proj** (`W_in`) is edited
- `layer_ranges_out` — layers where **down-proj** (`W_out`) is edited

Reproduce configs pin one span per side, together with single `in_deltas` / `out_deltas` values. Example on Gemma (Ancient Rome):

```yaml
in_deltas: [7.0]
out_deltas: [1.0]
layer_ranges_in: [[0, 8]]
layer_ranges_out: [[13, 25]]
```

Pinned layer ranges and RMU update settings are validated against the allowed grid presets for each model at config load time.
