# Experiment Results (Pre-Defense Baseline)

Updated: 2026-04-07

## Scope

This document records baseline observations for the 3 main demo flows:
1. Segmentation
2. Inpainting
3. Generation

Notes:
- Current numbers are baseline ranges from manual validation and implementation logs.
- Replace with your final 5-10 image measurements before final defense.

## Test Setup

- Backend: FastAPI on WSL/Linux
- Frontend: Flutter app on Android test device
- GPU: GTX 1650 4GB
- Async inpainting job store: Redis

## Dataset Protocol (for final run)

Use 10 room images:
- 4 bright rooms
- 3 medium-light rooms
- 3 low-light rooms

For each image:
1. Segment one object (chair/sofa/table)
2. Remove object (inpainting)
3. Run one generation style
4. Record latency, success/fail, and quality notes

## Baseline Latency (Current)

| Pipeline Step | Method | Observed Range | Source |
|---|---|---|---|
| Segmentation | Local SAM | ~2-3s | Prior manual tests + progress notes |
| Segmentation | SAM3 cloud | variable (network dependent) | Current implementation behavior |
| Inpainting | LaMa (Replicate) | fast (few seconds class) | Service config + manual observation |
| Inpainting | SD Replicate | ~10s class | Service comments and API behavior |
| Inpainting | Local SD | very slow (minutes) | GTX 1650 limitation |
| Generation | ControlNet style generation | variable by style | Frontend/manual observation |

## Quality Observations (Current)

### Segmentation

- Local SAM point mode is stable for MVP demo.
- SAM3 path currently relies on text-guided behavior; user prompt quality impacts result.

### Inpainting

- LaMa is strongest for clean object removal without hallucinating new furniture.
- Some outputs still look soft in removed region, especially with broad masks.
- Better quality is seen when mask quality is high and object boundary is clean.

### Generation

- Style generation works end-to-end.
- Runtime varies by style/prompt complexity and network conditions.

## Success Rate Template (Fill before defense)

| Flow | Attempts | Success | Success Rate |
|---|---:|---:|---:|
| Segmentation | 10 | TBD | TBD |
| Inpainting | 10 | TBD | TBD |
| Generation | 10 | TBD | TBD |
| Full end-to-end | 10 | TBD | TBD |

## Cost Notes

- Local compute paths: no direct API cost, but high latency on 4GB VRAM.
- Cloud paths (Replicate): lower latency, per-image API cost.

## Action Items Before Final Defense

1. Run full 10-image benchmark and replace all TBD values.
2. Add one chart for latency and one chart for success rate.
3. Freeze 3 best demo examples with before/after screenshots.
