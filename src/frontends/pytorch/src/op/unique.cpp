// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/unique.hpp"

#include <memory>

#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unique.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {
using namespace ov::op;

Output<Node> flip(const NodeContext& context, const Output<Node>& x, const Output<Node>& axis) {
    auto minus_one = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    auto minimum_int =
        context.mark_node(v0::Constant::create(element::i32, Shape{1}, {std::numeric_limits<int>::min()}));
    auto slice = context.mark_node(std::make_shared<v8::Slice>(x, minus_one, minimum_int, minus_one, axis));
    return slice;
};

OutputVector translate_unique2(const NodeContext& context) {
    // torch.unique(input, sorted=True, return_inverse=False, return_counts=False) →
    // Tuple[Tensor, Tensor, Tensor]
    num_inputs_check(context, 1, 4);
    auto x = context.get_input(0);
    auto const_empty = std::make_shared<v0::Constant>(element::i64, Shape{0}, std::vector<int64_t>{});
    auto const_zero = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));

    bool sorted = true;
    bool return_inverse = false;
    bool return_counts = false;
    if (!context.input_is_none(1)) {
        sorted = context.const_input<bool>(1);
    }
    if (!context.input_is_none(2)) {
        return_inverse = context.const_input<bool>(2);
    }
    if (!context.input_is_none(3)) {
        return_counts = context.const_input<bool>(3);
    }

    OutputVector result;
    auto outputs = context.mark_node(std::make_shared<v10::Unique>(x, sorted));
    auto unique_values = outputs->output(0);
    if (!sorted) {
        // When not sorted, the default order should be flipped to match PyTorch implementation.
        unique_values = flip(context, unique_values, const_zero);
    }
    result.push_back(unique_values);
    if (return_inverse) {
        auto x_shape = context.mark_node(std::make_shared<v0::ShapeOf>(x));
        auto inverse = context.mark_node(std::make_shared<v1::Reshape>(outputs->output(2), x_shape, false));

        if (!sorted) {
            auto unique_values_shape = context.mark_node(std::make_shared<v0::ShapeOf>(unique_values));
            auto dim = context.mark_node(std::make_shared<v8::Gather>(unique_values_shape, const_zero, const_zero));
            auto broadcasted_dim = context.mark_node(std::make_shared<v3::Broadcast>(dim, x_shape));
            inverse = context.mark_node(std::make_shared<v1::Subtract>(broadcasted_dim, inverse));
            auto const_one = context.mark_node(v0::Constant::create(inverse->get_element_type(), Shape{1}, {1}));
            inverse = context.mark_node(std::make_shared<v1::Subtract>(inverse, const_one));
        }
        result.push_back(inverse);
    } else {
        result.push_back(const_empty);
    }
    if (return_counts) {
        auto counts = outputs->output(3);

        if (!sorted) {
            counts = flip(context, counts, const_zero);
        }
        result.push_back(counts);
    } else {
        result.push_back(const_empty);
    }

    return result;
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
