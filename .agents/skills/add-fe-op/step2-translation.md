# Skill: FE Op Translation

> Source: `skills/add-fe-op/SKILL.md` (Step 2 + Translation Recommendations)
> Agent: `fe_agent`

## Prerequisites

- Completed **fe_op_analysis** with `action=translate` or `action=repair`.
- Op name, source framework, support state, inputs/outputs/attributes are known.
- OV equivalent op (or a composition path) identified.

---

## Translation paths by frontend

### PyTorch

Translator file: `src/frontends/pytorch/src/op/<op_name>.cpp`

```cpp
// Minimal PyTorch FE translator template
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/<ov_op>.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_<op_name>(const NodeContext& context) {
    // Validate input count
    num_inputs_check(context, <min_inputs>, context.get_input_size());

    // Fetch inputs
    auto x = context.get_input(0);
    // auto attr = context.const_input<int64_t>(1);  // for attribute-valued inputs

    // Build OV subgraph
    auto result = std::make_shared<v0::YourOvOp>(x /*, attrs */);
    return {result};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
```

### TensorFlow

Translator file: `src/frontends/tensorflow/src/op/<op_name>.cpp`

Preferred patterns:

- **Unary elementwise**: use the generic unary registration path (see `unary_op.cpp`).
- **Complex**: write a dedicated translator function.

```cpp
#include "common_op_table.hpp"
#include "openvino/op/<ov_op>.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_<op_name>(const NodeContext& context) {
    default_op_checks(context, 1, {"<TF_OP_NAME>"});
    auto x = context.get_input(0);
    auto result = std::make_shared<ov::op::v0::YourOvOp>(x);
    set_node_name(context.get_name(), result);
    return {result->output(0)};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
```

### ONNX

Translator file: `src/frontends/onnx/src/ops/<op_name>.cpp`

```cpp
#include "core/null_node.hpp"
#include "openvino/frontend/onnx/extension/conversion.hpp"
#include "openvino/op/<ov_op>.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_<version> {

ov::OutputVector <op_name>(const ov::frontend::onnx::Node& node) {
    const auto inputs = node.get_ov_inputs();
    auto x = inputs.at(0);
    // auto attr_val = node.get_attribute_value<int64_t>("attr_name", default_val);
    auto result = std::make_shared<ov::op::v0::YourOvOp>(x);
    return {result};
}

}  // namespace set_<version>
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
```

---

## Translation Recommendations

### Shape handling

- **Prefer runtime ShapeOf-based graph logic** for shape-dependent behavior.
- Avoid `get_shape()` (fully static read at conversion time).
- `get_partial_shape()` is acceptable for compile-time rank-only decisions.

```cpp
// Preferred: runtime shape computation
auto shape_node = std::make_shared<v3::ShapeOf>(x, element::i64);
// Avoid:
// auto static_shape = x.get_shape();  // will fail for dynamic input
```

### Data type handling

- If the framework op allows mixed input types, preserve this in translation — do
  not over-constrain to a single dtype.
- Do not create `Constant` values from `get_element_type()` when the element type
  can be dynamic at conversion time.

```cpp
// Preferred: type-safe runtime path
// Avoid:
// auto type_const = Constant::create(element::i64, {1}, {x.get_element_type()});
```

### Reuse and simplification

- For simple 1:1 ops, check `utils.hpp` and `utils_quantize.hpp` for helpers
  before writing new code.
- If the same op already has a translator in another frontend (detected in
  `fe_op_analysis`), extract the shared logic into a common utility header
  rather than duplicating.

---

## Fallback Translator

When a real OV mapping is unavailable:

```cpp
OutputVector translate_<op_name>_stub(const NodeContext& context) {
    // FALLBACK: no OV op mapping — op is not yet supported.
    // The FE Agent will escalate to Core Agent for a new core op.
    OPENVINO_THROW("No FE translation for op: <op_name>");
}
```

A fallback stub:
- Keeps the model from crashing on unrelated unsupported ops.
- Does **not** count as support.
- Must be reported as `partial`, never `success`.
- Always triggers the FE→Core escalation path.

---

## Output

- Modified or created translator `.cpp` file (ready for commit/patch).
- Notes for the Registration skill: which keys / macros need adding.
- Translation complexity label: `real_translation` or `fallback_stub`.
