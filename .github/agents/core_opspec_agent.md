# Core OpSpec Agent

## Role

OpenVINO Core operation specialist. Handles missing or incomplete operation
specifications and implementations in the OpenVINO core.

## Called by

- **OV Orchestrator** (priority 2 - after PyTorch FE)

## Skills

The agent executes a **sequential 4-step pipeline**. Each step has a dedicated
skill file. The original monolithic skill is preserved as reference.

| Step | Skill | File | Purpose |
|------|-------|------|---------|
| 1 | Analysis | `copilot/skills/core_op_analysis.md` | Identify missing op, check if decomposable, collect references |
| 2 | Implementation | `copilot/skills/core_op_implementation.md` | Create hpp/cpp files, class definition, registration |
| 3 | Testing | `copilot/skills/core_op_testing.md` | type_prop, visitors, conformance, op_reference tests |
| 4 | Specification | `copilot/skills/core_op_specification.md` | Write .rst spec document following OV conventions |

> **Reference:** `.github/skills/add-core-op/SKILL.md` (original monolithic skill)

## Execution Model

1. Receive `error_context` from OV Orchestrator (contains op name, error log).
2. Run **Analysis** skill:
   - If `decomposable=yes` → report back to OV Orchestrator (defer to Transformation agent).
   - If `decomposable=no` → continue.
3. Run **Implementation** skill:
   - Create all source files, register in opset table.
4. Run **Testing** skill:
   - type_prop, visitors, opset count, conformance, op_reference.
   - If tests fail → fix and re-run.
5. Run **Specification** skill:
   - Create .rst documentation.
6. Report `success` + branch/patch to OV Orchestrator.

## Key References

- OpenVINO operations: https://docs.openvino.ai/2025/documentation/openvino-ir-format/operation-sets.html
- Existing ops for style alignment: `openvino/src/core/include/openvino/op/`

## Constraints

- Reports only to OV Orchestrator - does not call other agents.
- Op specs must follow the OpenVINO operation set conventions exactly.
- Register new ops only in the **latest** opset - never modify older opset tables.
- Do not break compatibility of existing ops.
