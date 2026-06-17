# Mitsuba memory leak — findings & fix

## Problem
The training memory leak is host RSS (~2.15 MB/step), not VRAM. Root cause:
`_render_observation` calls `mi.load_dict()` every step, and glibc/Dr.Jit don't
fully return the freed memory. Junyi's periodic flush only cuts the rate to ~1/4
— still linear, still OOMs before convergence.

## Fix
Load the scene once with `optimize=False` (disables Mitsuba's mesh merging so
each mesh stays individually addressable), then per step only update
`vertex_positions` + light/camera `to_world` and call `params.update()` — no more
`load_dict`.

Implemented as three new methods on `MitsubaRenderer`
(`build_persistent_scene` / `update_persistent_scene` / vertex cache), supporting
multiple integrators (path + depth) sharing one vertex cache. All three
`_render_observation*` methods in `lumi_env.py` now use the persistent path.
`mitsuba_flush_interval` defaulted to 0 (flush no longer needed).

## Verification
- **Vertices**: persistent path matches the old path bit-for-bit (max |Δ| 0).
- **Render**: only difference vs. old path is Monte Carlo sampling noise that
  falls as 1/√spp (0.50× from 16→64 spp, textbook) — zero-mean, no bias. Lights
  and light_diff intact.
- **Leak (render only)**: RSS goes from +2.15 MB/step → **+0.00 MB/step** after
  warmup. light_diff and rgbd both confirmed flat.
- **Real training**: ran 12k steps (light_diff). No Mitsuba-style leak — RSS
  growth rate decays and shows large GC sawtooth (not the smooth monotonic rise
  of a leak); residual slow growth is the DrQ-v2 buffer workers + CUDA cache
  (bounded, present in original code too). Agent learns normally (episode reward
  ~0.03 → 3–8), confirming the optimize=False sampling noise does not hurt
  training.

## Summary
| | per-step leak | result |
|---|---|---|
| original | +2.15 MB | OOM |
| Junyi's flush | +0.52 MB | still OOM |
| persistent scene | +0.00 MB | trains indefinitely |