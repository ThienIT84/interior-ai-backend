# Trade-offs and Limitations

Updated: 2026-04-07

## 1) Inpainting Backend Strategy

Decision:
- Use hybrid chain: lama -> replicate -> local

Why:
- LaMa is fast and works well for object removal.
- Replicate SD is useful fallback when needed.
- Local SD on GTX 1650 is too slow for stable demo flow.

Trade-off:
- Better demo reliability and speed, but cloud dependency and API cost remain.

## 2) Async Job Persistence Scope

Decision:
- Persist async inpainting jobs in Redis.
- Generation/placement jobs are currently in-memory.

Why:
- Inpainting flow was prioritized for reliability first.
- Generation persistence was deferred to keep MVP scope manageable.

Trade-off:
- Backend restart can lose generation job state.
- Not production-ready for long-running generation workloads.

## 3) Segmentation Backend Choice (Local vs Cloud)

Decision:
- Keep both local SAM and SAM3 cloud modes.

Why:
- Local SAM gives predictable on-device behavior.
- SAM3 cloud can improve flexibility for text-guided segmentation.

Trade-off:
- Cloud mode is network/token dependent.
- Current SAM3 behavior does not fully map to classic point-prompt semantics.

## 4) Quality vs Speed in Inpainting

Decision:
- Prioritize clean removal and stable outputs over fully generative replacement.

Why:
- For this project, objective is removing old furniture first.

Trade-off:
- Some removed regions can look soft.
- Additional sharpening/refinement may be needed for high-detail outputs.

## 5) Hardware Constraints (GTX 1650 4GB)

Observed constraints:
- Limited VRAM for heavy local diffusion workflows.
- Slower local generation/inpainting under full-quality settings.

Impact:
- Hybrid cloud/local architecture is required for practical demo latency.

## 6) Testing Scope

Current status:
- Smoke tests exist for health, segmentation contract, and inpainting async contract.
- Flutter smoke test exists for main screen rendering.

Limitation:
- Coverage is still baseline, not full regression suite.

## 7) MVP vs Production-Ready Boundary

MVP-ready:
- Segmentation -> Inpainting -> Generation end-to-end demo flow.

Not production-ready yet:
- Full persistent job management for all generation flows.
- Comprehensive benchmark/observability and failure-recovery hardening.
- AR module implementation.

## 8) Recommended Next Iteration

1. Move generation job state to Redis.
2. Add final benchmark table (10-image run) and charts.
3. Add retry/backoff strategy for cloud calls.
4. Expand test suite with integration and regression cases.
