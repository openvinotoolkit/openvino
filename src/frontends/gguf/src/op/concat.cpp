// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include "openvino/frontend/exception.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"

#include "node_context.hpp"
#include "op_table.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

// GGML_OP_CONCAT: concatenate two inputs along a ggml axis. The decoder exposes the raw ggml
// dimension index as "concat_axis"; here it is converted to the reversed OV axis order.
OutputVector translate_concat(const NodeContext& context) {
    num_inputs_check(context, 2, 2);

    const auto output_shape = context.get_output_shape();
    FRONT_END_OP_CONVERSION_CHECK(output_shape.rank().is_static(), "CONCAT requires static output rank");
    const auto rank = output_shape.rank().get_length();

    const int ggml_dim = context.get_attribute<int>("concat_axis");
    FRONT_END_OP_CONVERSION_CHECK(ggml_dim >= 0 && ggml_dim < rank, "CONCAT axis is out of range");

    auto input_0 = context.get_input(0);
    auto input_1 = context.get_input(1);
    const auto output_type = context.get_attribute<ov::element::Type>("output_type");

    if (input_0.get_element_type() != output_type) {
        input_0 = std::make_shared<ov::op::v0::Convert>(input_0, output_type);
    }
    if (input_1.get_element_type() != output_type) {
        input_1 = std::make_shared<ov::op::v0::Convert>(input_1, output_type);
    }

    const auto axis = static_cast<int64_t>(rank - 1 - ggml_dim);
    auto res = std::make_shared<ov::op::v0::Concat>(OutputVector{input_0, input_1}, axis);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
