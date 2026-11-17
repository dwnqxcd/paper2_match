# Cross-modal UAV registration: evaluation code

Code and per-frame results for

> **Scale Normalization, Correspondence Sampling Budget and Risk-Coverage-Gated Verification
> for Cross-Modal UAV Registration**

The pipeline has three stages, each evaluated on its own and then cumulatively:

1. **Scale normalization** — the query is rescaled by the platform GSD ratio when the reference is a
   georeferenced chip, or the reference is downscaled to the query width when it is a second camera image.
2. **Correspondence sampling** — a fixed or GSD-derived budget is drawn from the dense warp field, either
   with the matcher's own sampler or with an 8x8 grid-stratified rule.
3. **Verification** — inlier count, inlier occupancy on the grid, and the median inlier residual decide
   whether a transform is accepted, escalated, or rejected.

## Data

The M3T-UAV benchmark is not redistributed here. Download it from
<https://github.com/JinHaiXu0322/M3T-UAV> and place the test split at

```
data/M3T/test_dataset/
    ir.json  vis.json
    test01/ ... test06/
        uav_ir/  uav_vis/  map/
```

Scripts skip frames whose satellite chip is missing or truncated, which is why the map tasks are
evaluated on 1,598 of the 1,602 frames.

## Environment

Python 3.13, PyTorch 2.6 (CUDA 12.4), OpenCV 5.0, Kornia 0.8, `romatch` for RoMa, plus the released
weights of SIFT (OpenCV), LoFTR, DISK, DeDoDe, XoFTR, MINIMA and CFOG. See `requirements.txt`.
All timings in the paper are wall-clock on one RTX 4060 Ti (8 GB) with an Intel i7-13700K.

## What produces what

| Result | Script |
|---|---|
| Table 2, Figure 3, Figure 4 (matcher comparison) | `src/run_m3t_crossmodal.py`, `src/run_m3t_cfog.py`, then `src/compute_all_method_stats.py` |
| Table 3, Table 4, Figure 5, Figure 6 (GSD normalization, per scene) | `src/run_map_baselines.py`, then `src/compute_map_stats.py` |
| Table 5, Figure 7 (sampling budget) | `src/run_density_full.py` |
| Draw-to-draw variability of the budget study | `src/run_density_repeats.py` |
| Table 6, Figure 9, Figure 10 (ablation, gate) | `src/run_pipeline_ablation.py` |
| Table 7, Figure 12 (oracle alignment) | `src/run_oracle_sampling.py` |
| Scene structure correlate (Section 4) | `src/measure_scene_structure.py` |
| Figures | `draw/*.py` |

Typical order:

```bash
python src/run_m3t_crossmodal.py --task tir_vis --methods sift,loftr,roma
python src/run_map_baselines.py --task vis_map --gsd 1
python src/run_density_full.py
python src/run_pipeline_ablation.py
python src/run_oracle_sampling.py
python src/compute_all_method_stats.py
python src/compute_map_stats.py
```

## Results included

`results/raw/` holds the per-frame tables behind every number in the paper, plus the summaries the
manuscript is generated from:

- `pipeline_ablation_full.csv` — per frame: corner error at each ablation stage, inlier count,
  occupancy, median residual, accept flag.
- `density_full.csv` — per frame and budget: candidates, inliers, residual, corner error, runtime.
- `density_repeats.csv` — repeated draws from the same warp field, for the variability estimate.
- `oracle_sampling_full.csv` — per frame: raw/stratified error and the oracle-corrected errors.
- `map_*_gsd{0,1}.csv` — per frame and matcher on the two map tasks.
- `*_stats.json` — the aggregates used by the tables.

Corner errors are reported in native query pixels, measured after dehomogenization, as defined in
Eq. 2 of the paper. They are unbounded by the image size: a homography fitted to a one-dimensional
inlier set extrapolates to 10^4 px, which is why the figures use a logarithmic axis.

## License

Code released under the MIT license. The benchmark keeps its own license.
