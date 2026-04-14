---
name: core-opset-initialization
description: Initializes a new opset (operation set) in OpenVINO. Use when starting development of a new opset version (e.g., opset17, opset18) before adding any new operations.
---

# Skill: core-opset-initialization

> **When to invoke:** Run this step **only when the target opset does not yet exist**
> in `/tmp/openvino`. If `opsetX.hpp` already exists, skip to
> `skills/add-core-op/step1-analysis.md`.

## When This Skill Applies
Use this skill when:
- A new opset version needs to be initialized in OpenVINO
- Starting development cycle for the next opset (e.g., transitioning from opset16 to opset17)

## Overview
An opset initialization creates the scaffolding for a new operation set version. This is done at the **beginning** of a new opset development cycle. The new opset starts with minimal operators (Parameter, Convert, ShapeOf) and full operator coverage is added at the **end** of development.

**Placeholder notation used in this document**
- `opsetX`: replace `X` with the concrete opset version number (for example, `opset17`, `opset18`).
- `X`: stands for the opset version number and must be replaced consistently in all filenames, namespaces, and comments.
- `<PREV>`: replace with the immediately preceding opset version number (for example, if `X` is 18, then `<PREV>` is 17).

## Files to Create

### 1. C++ Core Header Files

`**/openvino/src/core/include/openvino/opsets/opsetX.hpp`
```cpp
#pragma once

#include "openvino/op/ops.hpp"

namespace ov {
namespace opsetX {
#define _OPENVINO_OP_REG(a, b) using b::a;
#include "openvino/opsets/opsetX_tbl.hpp"
#undef _OPENVINO_OP_REG
}  // namespace opsetX
}  // namespace ov
```

`**/openvino/src/core/include/openvino/opsets/opsetX_tbl.hpp`
```cpp
#ifndef _OPENVINO_OP_REG
#    warning "_OPENVINO_OP_REG not defined"
#    define _OPENVINO_OP_REG(x, y)
#endif

// Previous opsets operators
// TODO (ticket: XXXXX): Add remaining operators from previous opset at the end of opsetX development
_OPENVINO_OP_REG(Parameter, ov::op::v0)
_OPENVINO_OP_REG(Convert, ov::op::v0)
_OPENVINO_OP_REG(ShapeOf, ov::op::v3)

// New operations added in opsetX
```

`**/openvino/src/core/dev_api/openvino/opsets/opsetX_decl.hpp`
```cpp
#pragma once

#include "openvino/op/ops_decl.hpp"

namespace ov::opsetX {
#define _OPENVINO_OP_REG(a, b) using b::a;
#include "openvino/opsets/opsetX_tbl.hpp"
#undef _OPENVINO_OP_REG
}  // namespace ov::opsetX
```

### 2. Python Binding Files

`**/openvino/src/bindings/python/src/openvino/opsetX/__init__.py`
```python
# New operations added in OpsetX

# Operators from previous opsets
# TODO (ticket: XXXXX): Add previous opset operators at the end of opsetX development
```

**IMPORTANT**: Do NOT add operator imports here at initialization. They are added during and at the END of opset development.

`**/openvino/src/bindings/python/src/openvino/opsetX/ops.py`
```python
"""Factory functions for ops added to openvino opsetX."""
from functools import partial

from openvino.utils.node_factory import _get_node_factory

_get_node_factory_opsetX = partial(_get_node_factory, "opsetX")

# -------------------------------------------- ops ------------------------------------------------
```

## Files to Update

### 1. C++ Core Registration

`**/openvino/src/core/include/openvino/opsets/opset.hpp`
Add declaration for `get_opsetX()`:
```cpp
/**
 * @brief Returns opsetX
 * @ingroup ov_opset_cpp_api
 */
const OPENVINO_API OpSet& get_opsetX();
```

`**/openvino/src/core/src/opsets/opset.cpp`
- Add `_OPENVINO_REG_OPSET(opsetX)` to the opset_map
- Add `get_opsetX()` implementation:
```cpp
const ov::OpSet& ov::get_opsetX() {
    static OpSet opset;
    static std::once_flag flag;
    std::call_once(flag, [&]() {
#define _OPENVINO_OP_REG(NAME, NAMESPACE) opset.insert<NAMESPACE::NAME>();
#include "openvino/opsets/opsetX_tbl.hpp"
#undef _OPENVINO_OP_REG
    });
    return opset;
}
```

### 2. Test Files

`**/openvino/src/core/tests/op.cpp`
Add: `doTest(ov::get_opsetX);`

`**/openvino/src/core/tests/opset.cpp`
- Add include: `#include "openvino/opsets/opsetX.hpp"`
- Add test params: `OpsetTestParams{ov::get_opsetX, 3}` (3 = initial operator count)

`**/openvino/src/bindings/python/tests/test_transformations/test_pattern_ops.py`
Update: `last_opset_number = X`

### 3. Plugin Includes

`**/openvino/src/plugins/template/src/plugin.cpp`
Add: `#include "openvino/opsets/opsetX_tbl.hpp"`

`**/openvino/src/tests/functional/plugin/conformance/test_runner/op_conformance_runner/src/op_impl_check/single_op_graph.cpp`
Add: `#include "openvino/opsets/opsetX_tbl.hpp"`

### 4. Documentation

`**/openvino/docs/sphinx_setup/api/ie_python_api/api.rst`
Add autosummary entry for `openvino.opsetX`

### 5. Python Package Init (3 files must stay aligned!)

`**/openvino/src/bindings/python/src/openvino/__init__.py`
`**/openvino/tools/ovc/openvino/__init__.py`
`**/openvino/tools/benchmark_tool/openvino/__init__.py`

Add to ALL THREE files: `from openvino import opsetX`
CMake verifies these files are identical at build time. If they differ, the build fails.

## Files NOT to Update at Initialization

### Frontend Extension (DO NOT CHANGE)
`**/openvino/src/frontends/common/include/openvino/frontend/extension/op.hpp`

The "latest" opset reference should remain pointing to the **previous** opset:
```cpp
// Keep as-is during initialization - do not change to opsetX:
return ov::get_opset<PREV>();  // TODO: Update to opsetX at the end of opsetX development
```

This is updated only at the **END** of opset development when it becomes stable.

## Summary Checklist

| Action | File | Notes |
|--------|------|-------|
| CREATE | `**/openvino/src/core/include/openvino/opsets/opsetX.hpp` | Namespace header |
| CREATE | `**/openvino/src/core/include/openvino/opsets/opsetX_tbl.hpp` | Op table (minimal: 3 ops) |
| CREATE | `**/openvino/src/core/dev_api/openvino/opsets/opsetX_decl.hpp` | Dev API declaration |
| CREATE | `**/openvino/src/bindings/python/src/openvino/opsetX/__init__.py` | **Empty** — no imports |
| CREATE | `**/openvino/src/bindings/python/src/openvino/opsetX/ops.py` | Factory stub |
| UPDATE | `**/openvino/src/core/include/openvino/opsets/opset.hpp` | Add get_opsetX() declaration |
| UPDATE | `**/openvino/src/core/src/opsets/opset.cpp` | Add registration + implementation |
| UPDATE | `**/openvino/src/core/tests/op.cpp` | Add doTest() |
| UPDATE | `**/openvino/src/core/tests/opset.cpp` | Add include + test params |
| UPDATE | `**/openvino/src/plugins/template/src/plugin.cpp` | Add tbl include |
| UPDATE | `**/openvino/src/tests/.../single_op_graph.cpp` | Add tbl include |
| UPDATE | `**/openvino/docs/sphinx_setup/api/ie_python_api/api.rst` | Add docs entry |
| UPDATE | `**/openvino/src/bindings/python/tests/.../test_pattern_ops.py` | Update last_opset_number |
| UPDATE | `**/openvino/src/bindings/python/src/openvino/__init__.py` | Add opsetX import |
| UPDATE | `**/openvino/tools/ovc/openvino/__init__.py` | Add opsetX import (must match above) |
| UPDATE | `**/openvino/tools/benchmark_tool/openvino/__init__.py` | Add opsetX import (must match above) |
| **SKIP** | `**/openvino/src/frontends/common/.../op.hpp` | Do NOT update "latest" |

## Common Mistakes to Avoid

1. **Python __init__.py with operator imports**: The opsetX/__init__.py should be EMPTY at initialization. Full operator imports are added at the END of development.

2. **Updating frontend "latest" opset**: Do NOT change op.hpp to return the new opset. It stays at the previous version until development is complete.

3. **Wrong initial op count in tests**: The opset.cpp test should use `3` as the initial operator count (Parameter, Convert, ShapeOf).

4. **Misaligned Python __init__.py files**: Three `__init__.py` files must be identical — CMake checks alignment at build time. Update all three together.
