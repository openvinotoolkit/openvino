# Skill: FE Op Registration — Verification Checklist

> For registration instructions see the per-frontend skill files:
> - **PyTorch**: [pytorch.md](pytorch.md) §4 and §5 (op_table.cpp, TorchScript key, FX key)
> - **ONNX**: [onnx.md](onnx.md) §4 (ONNX_OP macro in per-op `.cpp`)
> - **TensorFlow / generic**: [SKILL.md](SKILL.md) §3

Use this checklist to confirm registration is complete before proceeding to testing.

---

## Verification Checklist

**PyTorch:**
- [ ] `OP_CONVERTER(translate_<op_name>)` declaration present in `op_table.cpp`
- [ ] TorchScript key `aten::<op_name>` registered
- [ ] FX key `aten.<op_name>.default` registered
- [ ] All overload variants covered (`aten.<op_name>.Tensor`, `aten.<op_name>.Scalar`, …)
- [ ] `CMakeLists.txt` updated to include new `.cpp`

**TensorFlow:**
- [ ] `op_table.cpp` entry present (unary path or dedicated)
- [ ] `CMakeLists.txt` updated

**ONNX:**
- [ ] `ONNX_OP` macro present in translator `.cpp` file (not in `ops_bridge.cpp`)
- [ ] Opset range is accurate (cross-check with ONNX spec)
- [ ] `CMakeLists.txt` updated

**All frontends:**
- [ ] No duplicate registrations introduced
