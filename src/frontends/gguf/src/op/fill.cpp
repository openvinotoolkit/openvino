// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <memory>
#include <vector>

#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/shape_of.hpp"

#include "node_context.hpp"
#include "op_table.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

// GGML_OP_FILL sets every element of a tensor to a constant. The scalar is provided by the decoder
// as the "fill_value" attribute (ggml stores it as a float in op_params[0]). The output has the same
// shape as the (single) input; broadcasting to ShapeOf(input) keeps the input live and works for
// dynamic shapes.
OutputVector translate_fill(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    auto x = context.get_input(0);
    float fill_value = context.get_attribute<float>("fill_value");

    ov::Output<ov::Node> val = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {fill_value});
    // Emit the fill scalar in the output type so the result matches the ggml tensor's dtype (the
    // broadcast target keeps the shape; the type comes from the scalar).
    const auto output_type = context.get_attribute<ov::element::Type>("output_type");
    if (output_type.is_static() && output_type != ov::element::f32) {
        val = std::make_shared<ov::op::v0::Convert>(val, output_type);
    }
    auto target_shape = std::make_shared<ov::op::v3::ShapeOf>(x, ov::element::i64);
    auto res = std::make_shared<ov::op::v3::Broadcast>(val, target_shape);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
