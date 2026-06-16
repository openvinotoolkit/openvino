# Skill: FE Op Testing

> Source: `skills/add-fe-op/SKILL.md` (Steps 4, 5, 6)

## Prerequisites

- Completed **fe_op_registration** — translator file and registration entries are in place.

---

## Writing Tests

Follow the test patterns described in the per-frontend skill files:
- **PyTorch**: [pytorch.md](pytorch.md) §5 — `PytorchLayerTest` class structure, TorchScript and FX paths, `@pytest.mark.nightly` / `@pytest.mark.precommit`
- **ONNX**: [onnx.md](onnx.md) §6 — `.prototxt` test model and C++ GTest pattern
- **TensorFlow / generic**: follow naming conventions in `tests/layer_tests/tensorflow_tests/`

Look at tests for a similar existing op in the same directory as the starting point.

---

## Conversion Validation

After writing translator and registration, run the end-to-end conversion check using the
**`verify-conversion`** skill:

> Read and follow: `skills/verify-conversion/SKILL.md`

The skill auto-detects the conversion path (optimum-intel / ovc / convert_model) and
runs a numerical sanity check. Report the outcome (`validation=pass|fail|blocked`) back
to the FE Agent before generating the patch.

---

## Git Patch Generation

After translator, registration, and test files are written, generate a patch file.
Run:

```
python .github/scripts/meat/generate_patch.py --component fe_<frontend> --op <op_name>
```

Place the resulting `.patch` file in `agent-results/frontend/patches/`.

---

## Save Patch and Report

Write the patch path and validation status to the FE Agent output file:

```
python .github/scripts/meat/record_result.py --agent frontend --op <op_name> --patch <patch_path> --validation <pass|fail|blocked>
```

---

## Reporting States

| State | Meaning |
|-------|---------|
| `success` | Real translation in place, conversion produces OV graph nodes, tests written |
| `partial` | Fallback stub in place — model does not crash, but no real OV graph yet |
| `blocked` | Validation could not run (installed OV does not contain FE patch) |
| `failed` | Translator or test produced errors |

---

## Output

- Test files created and ready for commit.
- Git patch file: `patches/fe_<op_name>_<frontend>.patch`
- Patch available in `agent-results/pytorch-fe/patches/`.
- Validation status reported back to FE Agent.
