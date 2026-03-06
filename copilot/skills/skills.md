# Skills Registry

This file lists all available skills for Copilot agents.
Each skill is a self-contained procedural document that an agent loads and executes.

## Pipeline Helper Scripts

These scripts are called directly from workflow `run:` steps and replace fragile
inline shell/Python logic.  All live under `scripts/` and are independently testable.

| Script | File | Used by |
|--------|------|---------|
| Detect task | `scripts/detect_task.py` | Deployer, Optimum-Intel |
| Classify error | `scripts/classify_error.py` | Deployer (on failure) |
| Classify OV component | `scripts/classify_ov_component.py` | OV Orchestrator classify job |
| Parse WWB score | `scripts/parse_wwb_score.py` | WWB Benchmark |
| Post issue comment | `scripts/post_issue_comment.py` | All workflows (issue comments) |

## Optimum-Intel Skills

| Skill | File | Used by |
|-------|------|---------|
| Bootstrap | `copilot/skills/optimum_bootstrap.md` | Optimum-Intel Agent |
| Model Conversion | `copilot/skills/optimum_model_conversion.md` | Optimum-Intel Agent |
| Debug Export | `copilot/skills/optimum_debug_export.md` | Optimum-Intel Agent |
| Create Tiny Model | `copilot/skills/optimum_create_tiny_model.md` | Optimum-Intel Agent |
| Create Model Config | `copilot/skills/optimum_create_model_config.md` | Optimum-Intel Agent |
| Add Model Support | `copilot/skills/optimum_add_model_support.md` | Optimum-Intel Agent |

## OpenVINO Core Skills

| Skill | File | Used by |
|-------|------|---------|
| Op Analysis | `copilot/skills/core_op_analysis.md` | Core OpSpec Agent (Step 1) |
| Op Implementation | `copilot/skills/core_op_implementation.md` | Core OpSpec Agent (Step 2) |
| Op Testing | `copilot/skills/core_op_testing.md` | Core OpSpec Agent (Step 3) |
| Op Specification | `copilot/skills/core_op_specification.md` | Core OpSpec Agent (Step 4) |
| Add Core Op (ref) | `.github/skills/add-core-op/SKILL.md` | Core OpSpec Agent (original) |

## GPU Plugin Skills

| Skill | File | Used by |
|-------|------|---------|
| Hardware Analysis | `copilot/skills/gpu_hardware_analysis.md` | GPU Agent (Step 1) |
| Kernel Development | `copilot/skills/gpu_kernel_development.md` | GPU Agent (Step 2) |
| Performance Profiling | `copilot/skills/gpu_performance_profiling.md` | GPU Agent (Step 3) |
| Testing | `copilot/skills/gpu_testing.md` | GPU Agent (Step 4) |

## Enablement Guide Template

| Reference | Location |
|-----------|----------|
| Template | `docs/qwen3_5_35b_a3b_openvino_enablement.md` |
| Example PR | [PR #35](https://github.com/intel-sandbox/applications.ai.openvino.meat/pull/35) |
