// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "utils.hpp"

using namespace ov::op;

OutputVector translate_quantile(const NodeContext& context) {
    num_inputs_check(context, 2, 5);

    auto input = context.get_input(0);
    auto q = context.get_input(1);      // Quantile(s), can be float or tensor

    auto dim = context.input_is_none(2) ? -1 : context.get_input<int64_t>(2);
    auto keepdim = context.input_is_none(3) ? false : context.get_input<bool>(3);
    auto interpolation = context.input_is_none(4) ? "linear" : context.get_input<std::string>(4);


    if (dim == -1) {
        input = context.mark_node(std::make_shared<v0::Reshape>(
            input, context.mark_node(std::make_shared<v0::Range>(0, input.get_shape().size(), 1)), true));
        dim = 0;
    }

    auto sorted = context.mark_node(std::make_shared<v0::Sort>(input, dim, true)); // Ascending order

    auto dim_size = input.get_shape()[dim];

    auto indices = context.mark_node(std::make_shared<v0::Multiply>(q, dim_size - 1));
    auto lower_indices = context.mark_node(std::make_shared<v0::Floor>(indices));
    auto upper_indices = context.mark_node(std::make_shared<v1::Add>(lower_indices, 1));
    auto weights = context.mark_node(std::make_shared<v1::Subtract>(indices, lower_indices));
    auto lower_values = context.mark_node(std::make_shared<v1::Gather>(sorted, lower_indices, dim));
    auto upper_values = context.mark_node(std::make_shared<v1::Gather>(sorted, upper_indices, dim));

    Output<Node> result;
    if (interpolation == "linear") {
        result = context.mark_node(std::make_shared<v1::Add>(
            lower_values, context.mark_node(std::make_shared<v1::Multiply>(weights, upper_values))));
    } else if (interpolation == "lower") {
        result = lower_values;
    } else if (interpolation == "higher") {
        result = upper_values;
    } else if (interpolation == "nearest") {
        auto nearest_indices = context.mark_node(std::make_shared<v0::Round>(indices));
        result = context.mark_node(std::make_shared<v1::Gather>(sorted, nearest_indices, dim));
    } else if (interpolation == "midpoint") {
        result = context.mark_node(std::make_shared<v1::Add>(
            lower_values, context.mark_node(std::make_shared<v1::Multiply>(
                              context.mark_node(std::make_shared<v0::Constant>(element::f32, Shape{}, 0.5)),
                              context.mark_node(std::make_shared<v1::Subtract>(upper_values, lower_values))))));
    } else {
        throw std::runtime_error("Unsupported interpolation method: " + interpolation);
    }
    if (!keepdim) {
        auto reshape_dims = input.get_shape();
        reshape_dims.erase(reshape_dims.begin() + dim);
        result = context.mark_node(std::make_shared<v0::Reshape>(result, reshape_dims, true));
    }

    return {result};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
