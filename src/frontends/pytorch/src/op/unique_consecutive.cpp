// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/unique_consecutive.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {
using namespace ov::op;

OutputVector translate_unique_consecutive(const NodeContext& context) {
    // aten::unique_consecutive(Tensor self, bool return_inverse=False, bool return_counts=False, int dim=None) ->
    // (Tensor output, Tensor inverse_indices, Tensor counts)
    num_inputs_check(context, 1, 4);

    auto input = context.get_input(0);

    bool return_inverse = false;
    if (!context.input_is_none(1)) {
        return_inverse = context.const_input<bool>(1);
    }

    bool return_counts = false;
    if (!context.input_is_none(2)) {
        return_counts = context.const_input<bool>(2);
    }

    int64_t dim = -1;
    bool dim_is_none = context.input_is_none(3);
    if (!dim_is_none) {
        dim = context.const_input<int64_t>(3);
    }

    OutputVector outputs;
    Output<Node> prepared_input;
    Output<Node> axis_const;

    // Step 1: Choose the axis and prepare input
    if (dim_is_none) {
        // If dim is None, flatten the input tensor first
        auto shape = std::make_shared<v0::ShapeOf>(input);
        auto flatten_shape = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {-1}));
        prepared_input = context.mark_node(std::make_shared<v1::Reshape>(input, flatten_shape));
        // Use axis 0 for flattened tensor
        axis_const = context.mark_node(v0::Constant::create(element::i64, Shape{}, {0}))
    } else {
        // Use input as-is with specified dimension
        prepared_input = input;
        // Handle negative axis value
        if (dim < 0) {
            auto rank = context.mark_node(std::make_shared<v0::ShapeOf>(input));
            auto rank_scalar = context.mark_node(std::make_shared<v1::ReduceProd>(rank, v0::Constant::create(element::i64, Shape{}, {0})));
            auto dim_const = context.mark_node(v0::Constant::create(element::i64, Shape{}, {dim}));
            axis_const = context.mark_node(std::make_shared<v1::Add>(dim_const, rank_scalar));
        } else {
            axis_const = context.mark_node(v0::Constant::create(element::i64, Shape{}, {dim}));
        }
    }

    // Step 2: Compare the neighbors along axis a
    // build per-axis start/stop/step vectors (same length as input rank)
    auto shape = context.mark_node(std::make_shared<v0::ShapeOf>(prepared_input));
    auto zero_scalar = context.mark_node(v0::Constant::create(element::i64, Shape{}, {0}));
    auto one_scalar = context.mark_node(v0::Constant::create(element::i64, Shape{}, {1}));

    // broadcast scalars to shape-of-input to produce length-rank vectors
    auto zeros = context.mark_node(std::make_shared<v3::Broadcast>(zero_scalar, shape));
    auto ones = context.mark_node(std::make_shared<v3::Broadcast>(one_scalar, shape));
    
    // head: start = [0, ...], stop = shape - 1 (exclusive)
    auto head_start = zeros;
    auto head_stop = context.mark_node(std::make_shared<v1::Subtract>(shape, ones));
    auto step = ones;

    // tail: start = [1, ...], stop = shape (exclusive -> equals shape)
    auto tail_start = ones;
    auto tail_stop = shape;

    // slice along all axes; only the axis of interest changes values in the vectors above
    auto head = context.mark_node(std::make_shared<v8::Slice>(prepared_input, head_start, head_stop, step));
    auto tail = context.mark_node(std::make_shared<v8::Slice>(prepared_input, tail_start, tail_stop, step));

    auto equal = context.mark_node(std::make_shared<v1::Equal>(head, tail));

    // Build a keep mask of run starts

    // Get run start indices and the values output

    // Compute counts

    // Compute inverse indices
    
    outputs.push_back(unique_consecutive->output(0));
    if (return_inverse) {
        outputs.push_back(unique_consecutive->output(1));
    }
    if (return_counts) {
        outputs.push_back(unique_consecutive->output(return_inverse ? 2 : 1));
    }

    return outputs;
};
}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov