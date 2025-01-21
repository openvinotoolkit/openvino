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

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_quantile(const NodeContext& context) {
    num_inputs_check(context, 2, 4);

    auto input = context.get_input(0);
    auto quantiles = context.get_input(1);

    auto dim = context.input_is_none(2) ? -1 : context.get_input<int64_t>(2);
    auto keepdim = context.input_is_none(3) ? false : context.get_input<bool>(3);

    if (dim == -1) {
        input = context.mark_node(std::make_shared<v0::Reshape>(
            input, context.mark_node(v0::Constant::create(element::i64, {1}, {-1})), true));
        dim = 0;
    }

    auto sort_result = context.mark_node(std::make_shared<v0::Sort>(input, dim, true));
    auto sorted_tensor = sort_result->output(0);

    auto input_shape = context.mark_node(std::make_shared<v0::ShapeOf>(input));
    auto dim_size = context.mark_node(std::make_shared<v0::Gather>(
        input_shape, context.mark_node(v0::Constant::create(element::i64, {}, {dim})),
        v0::Constant::create(element::i64, {}, {0})));

    auto scaled_q = context.mark_node(std::make_shared<v1::Multiply>(
        quantiles, context.mark_node(std::make_shared<v1::Subtract>(
                       dim_size, v0::Constant::create(element::i64, {}, {1})))));
    auto lower_indices = context.mark_node(std::make_shared<v0::Floor>(scaled_q));
    auto upper_indices = context.mark_node(std::make_shared<v1::Add>(
        lower_indices, v0::Constant::create(element::i64, {}, {1})));

    lower_indices = context.mark_node(std::make_shared<v1::Maximum>(
        lower_indices, v0::Constant::create(element::i64, {}, {0})));
    upper_indices = context.mark_node(std::make_shared<v1::Minimum>(
        upper_indices, context.mark_node(std::make_shared<v1::Subtract>(
                           dim_size, v0::Constant::create(element::i64, {}, {1})))));

    auto lower_values = context.mark_node(std::make_shared<v1::Gather>(sorted_tensor, lower_indices, dim));
    auto upper_values = context.mark_node(std::make_shared<v1::Gather>(sorted_tensor, upper_indices, dim));

    auto weights = context.mark_node(std::make_shared<v1::Subtract>(scaled_q, lower_indices));

    auto result = context.mark_node(std::make_shared<v1::Add>(
        lower_values, context.mark_node(std::make_shared<v1::Multiply>(weights, context.mark_node(std::make_shared<v1::Subtract>(upper_values, lower_values))))));

    if (!keepdim) {
        auto input_shape = context.mark_node(std::make_shared<v0::ShapeOf>(input));
        auto output_shape = context.mark_node(std::make_shared<v1::Gather>(
            input_shape,
            context.mark_node(v0::Constant::create(element::i64, {1}, {dim})),
            v0::Constant::create(element::i64, {}, {0})));
        result = context.mark_node(std::make_shared<v0::Reshape>(result, output_shape, true));
    }

    return {result};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
