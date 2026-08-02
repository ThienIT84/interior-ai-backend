# MVP Benchmark Protocol

Use this protocol before demos or defense dry-runs.

## Dataset

Run exactly 10 room images:

| Bucket | Count | Notes |
|---|---:|---|
| Bright rooms | 4 | Clear walls/floors and visible furniture boundaries |
| Medium-light rooms | 3 | Normal indoor lighting |
| Low-light rooms | 3 | Shadows or weaker object boundaries |

## Measurements

For each image, record:

| Image | Bucket | Segmentation latency | Inpainting latency | Generation latency | Success | Error | Quality notes |
|---|---|---:|---:|---:|---|---|---|
| 01 | bright | TBD | TBD | TBD | TBD | TBD | TBD |
| 02 | bright | TBD | TBD | TBD | TBD | TBD | TBD |
| 03 | bright | TBD | TBD | TBD | TBD | TBD | TBD |
| 04 | bright | TBD | TBD | TBD | TBD | TBD | TBD |
| 05 | medium | TBD | TBD | TBD | TBD | TBD | TBD |
| 06 | medium | TBD | TBD | TBD | TBD | TBD | TBD |
| 07 | medium | TBD | TBD | TBD | TBD | TBD | TBD |
| 08 | low-light | TBD | TBD | TBD | TBD | TBD | TBD |
| 09 | low-light | TBD | TBD | TBD | TBD | TBD | TBD |
| 10 | low-light | TBD | TBD | TBD | TBD | TBD | TBD |

## Demo Acceptance

- Pick 3 happy-path examples from the 10-image run.
- Save original, mask overlay, inpainting result, and generation/placement result.
- Dry-run the demo twice: once with the fastest cloud path, once with the intended fallback path.
