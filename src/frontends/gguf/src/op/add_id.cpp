// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

namespace {
// Collapse a 4D [1, 1, a, b] input to 2D [a, b] when it isn't already 2D. Bias/id constants may
// already be stored 2D, hence the early-outs.
ov::Output<ov::Node> reshape_add_id_input_to_2d(const ov::Output<ov::Node>& input,
                                                const ov::PartialShape& input_shape,
                                                const std::vector<int>& dims) {
    const auto& actual_shape = input.get_partial_shape();
    if (actual_shape.rank().is_static() && actual_shape.rank().get_length() == 2) {
        return input;
    }
    if (input_shape.rank().is_static() && input_shape.rank().get_length() == 2) {
        return input;
    }
    auto shape = std::make_shared<ov::op::v3::ShapeOf>(input, ov::element::i64);
    auto target_shape = gather_dims(shape, dims);
    auto reshaped_input = std::make_shared<ov::op::v1::Reshape>(input, target_shape, false);
    return reshaped_input;
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

    ov::Output<ov::Node> safe_ids = ids;
    std::shared_ptr<ov::op::v1::GreaterEqual> valid_ids;
    if (!has_nonnegative_indices(ids)) {
        auto zero_id = ov::op::v0::Constant::create(ids.get_element_type(), ov::Shape{}, {0});
        valid_ids = std::make_shared<ov::op::v1::GreaterEqual>(ids, zero_id);
        safe_ids = std::make_shared<ov::op::v1::Select>(valid_ids, ids, zero_id);
    }
    auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    auto gathered_bias = std::make_shared<ov::op::v8::Gather>(bias, safe_ids, gather_axis);
    auto input_shape = std::make_shared<ov::op::v3::ShapeOf>(input, ov::element::i64);
    ov::Output<ov::Node> selected_bias = std::make_shared<ov::op::v1::Reshape>(gathered_bias, input_shape, false);

    if (selected_bias.get_element_type() != input.get_element_type()) {
        selected_bias = std::make_shared<ov::op::v0::Convert>(selected_bias, input.get_element_type());
    }

    auto added = std::make_shared<ov::op::v1::Add>(input, selected_bias);
    ov::Output<ov::Node> res = added;
    if (valid_ids) {
        auto embedding_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {-1});
        auto valid_rows = std::make_shared<ov::op::v0::Unsqueeze>(valid_ids, embedding_axis);
        res = std::make_shared<ov::op::v1::Select>(valid_rows, added, input);
    }

    return rename_outputs_with_suffix({std::move(res)}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
