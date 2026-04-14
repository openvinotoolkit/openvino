# Skill: FE Op Registration

> Source: `skills/add-fe-op/SKILL.md` (Step 3)
> Agent: `fe_agent`

## Prerequisites

- Completed **fe_op_translation** — translator `.cpp` file is written and ready.

---

## PyTorch Registration

File: `src/frontends/pytorch/src/op_table.cpp`

### Translator declaration (top-of-file block)

```cpp
// In the OP_CONVERTER declaration block:
OP_CONVERTER(translate_<op_name>);
```

### TorchScript key (`aten::` namespace)

```cpp
// In the inline PYTORCH_OP_SUPPORTED / op_map table:
{"aten::<op_name>", op::translate_<op_name>},
```

### FX key (`aten.` namespace, `.default` suffix)

```cpp
{"aten.<op_name>.default", op::translate_<op_name>},
```

Check whether overload variants are needed (e.g. `aten.<op_name>.Tensor`,
`aten.<op_name>.Scalar`). Grep the surrounding entries for the same op family:

```bash
grep -n 'aten\.<op_name>' src/frontends/pytorch/src/op_table.cpp
```

### PyTorch CMakeLists.txt

Add the new `.cpp` to `src/frontends/pytorch/CMakeLists.txt`:

```cmake
set(SOURCES
    ...
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/<op_name>.cpp
    ...)
```

---

## TensorFlow Registration

File: `src/frontends/tensorflow/src/op_table.cpp`

**Unary elementwise path** (preferred when applicable):
```cpp
REGISTER_UNARY_OP("<TF_OP_NAME>", translate_<op_name>);
```

**Dedicated path** (for non-unary or complex ops):
```cpp
{"<TF_OP_NAME>", CreatorFunction(translate_<op_name>)},
```

### TensorFlow CMakeLists.txt

Add to `src/frontends/tensorflow/CMakeLists.txt`.

---

## ONNX Registration

The `ONNX_OP` macro in the translator file covers both registration and opset
versioning. Ensure the macro is present at the bottom of the translator `.cpp`:

```cpp
ONNX_OP("<OpName>", OPSET_RANGE(<min_ver>, <max_ver>), op::set_<version>::<op_name>);
```

If the translator file already contains `ONNX_OP`, no separate registration
file changes are needed — just verify the opset range is correct.

### ONNX CMakeLists.txt

Add to `src/frontends/onnx/CMakeLists.txt` if the file is new.

---

## Verification Checklist

Run these checks before proceeding to the Testing skill:

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
- [ ] `ONNX_OP` macro present in translator file
- [ ] Opset range is accurate (cross-check with ONNX spec)
- [ ] `CMakeLists.txt` updated

**All frontends:**
- [ ] No duplicate registrations introduced
  ```bash
  grep -c 'aten::<op_name>' src/frontends/pytorch/src/op_table.cpp  # expect 1
  ```

---

## Output

- Updated `op_table.cpp` and `CMakeLists.txt` (ready to commit).
- Verification checklist completed and all items confirmed.
