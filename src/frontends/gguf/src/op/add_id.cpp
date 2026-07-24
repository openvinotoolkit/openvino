// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"

#include "node_context.hpp"
#include "op_table.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

namespace {
// Collapse a 4D [1, 1, a, b] input to 2D [a, b] when it isn't already 2D. Bias/id constants may
// already be stored 2D, hence the early-outs.
ov::Output<ov::Node> reshape_add_id_input_to_2d(const ov::Output<ov::Node>& input,
                                                const ov::PartialShape& input_shape,
                                                const std::vector<int>& dims) {
    const auto actual_shape = input.get_partial_shape();
    if (actual_shape.rank().is_static() && actual_shape.rank().get_length() == 2) {
        return input;
    }
    if (input_shape.rank().is_static() && input_shape.rank().get_length() == 2) {
        return input;
    }
    auto shape = std::make_shared<ov::op::v3::ShapeOf>(input, ov::element::i64);
    return std::make_shared<ov::op::v1::Reshape>(input, get_dimensions(shape, dims), false);
}
}  // namespace

// GGML_OP_ADD_ID: gather per-token bias rows selected by ids and add to the input (MoE bias).
OutputVector translate_add_id(const NodeContext& context) {
    num_inputs_check(context, 3, 3);

    auto input = context.get_input(0);
    auto bias = context.get_input(1);
    auto ids = context.get_input(2);

    // OpenVINO uses reversed GGML dimensions:
    //   input: [1, n_token, n_used, n_embd]
    //   bias:  [1, 1, n_expert, n_embd]
    //   ids:   [1, 1, n_token, n_used]
    bias = reshape_add_id_input_to_2d(bias, context.get_input_shape(1), {2, 3});
    ids = reshape_add_id_input_to_2d(ids, context.get_input_shape(2), {2, 3});

    if (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    ov::Output<ov::Node> selected_bias = std::make_shared<ov::op::v8::Gather>(bias, ids, gather_axis);
    selected_bias = std::make_shared<ov::op::v1::Reshape>(
        selected_bias, std::make_shared<ov::op::v3::ShapeOf>(input, ov::element::i64), false);

    if (selected_bias.get_element_type() != input.get_element_type()) {
        selected_bias = std::make_shared<ov::op::v0::Convert>(selected_bias, input.get_element_type());
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Add>(input, selected_bias);
    const auto output_type = context.get_attribute<ov::element::Type>("output_type");
    if (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
