// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/concat.hpp"

#include <limits>
#include <memory>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/convert.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

// GGML_OP_CONCAT: concatenate inputs along a ggml axis. The decoder exposes the raw ggml
// dimension index as "concat_axis"; here it is converted to the reversed OV axis order.
// Native builders may pass more than two inputs, which equals a chain of binary concats.
OutputVector translate_concat(const NodeContext& context) {
    num_inputs_check(context, 2, std::numeric_limits<size_t>::max());

    const auto output_shape = context.get_input(0).get_partial_shape();
    FRONT_END_OP_CONVERSION_CHECK(output_shape.rank().is_static(), "CONCAT requires static input rank");
    const auto rank = output_shape.rank().get_length();

    const int ggml_dim = context.get_attribute<int>("concat_axis");
    FRONT_END_OP_CONVERSION_CHECK(ggml_dim >= 0 && ggml_dim < rank, "CONCAT axis is out of range");

    const auto output_type = context.get_input(0).get_element_type();
    OutputVector inputs;
    for (size_t i = 0; i < context.get_input_size(); ++i) {
        auto input = context.get_input(static_cast<int>(i));
        if (input.get_element_type() != output_type) {
            input = std::make_shared<ov::op::v0::Convert>(input, output_type);
        }
        inputs.push_back(input);
    }

    const auto axis = static_cast<int64_t>(rank - 1 - ggml_dim);
    auto res = std::make_shared<ov::op::v0::Concat>(inputs, axis);

    return rename_outputs_with_suffix({std::move(res)}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
