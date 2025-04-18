
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_bincount(const NodeContext& context) {
    num_inputs_check(context, 3, 3);

    auto input = context.get_input(0);
    auto weights = context.get_input(1);
    auto minlength = context.get_input(2);

    auto input_int = context.mark_node(std::make_shared<v0::Convert>(input, element::i32));

    auto max_val = context.mark_node(
        std::make_shared<v1::ReduceMax>(input_int, v0::Constant::create(element::i32, Shape{}, {0}), true));
    auto max_length =
        context.mark_node(std::make_shared<v1::Add>(max_val, v0::Constant::create(element::i32, Shape{}, {1})));

    auto output_size = context.mark_node(
        std::make_shared<v1::Maximum>(max_length,
                                      context.mark_node(std::make_shared<v0::Convert>(minlength, element::i32))));

    auto output = context.mark_node(v0::Constant::create(element::f32, Shape{1}, {0}));
    output = context.mark_node(std::make_shared<v3::Broadcast>(output, output_size));

    auto weight_tensor = weights.get_node_shared_ptr() != nullptr
                             ? weights
                             : context.mark_node(v0::Constant::create(element::f32, input.get_shape(), {1}));

    auto reshaped_input = context.mark_node(
        std::make_shared<v1::Reshape>(input_int,
                                      v0::Constant::create(element::i32, Shape{2}, {input_int.get_shape()[0], 1}),
                                      true));

    auto result = context.mark_node(std::make_shared<v3::ScatterNDUpdate>(output,
                                                                          reshaped_input,
                                                                          weight_tensor,
                                                                          v3::ScatterNDUpdate::Reduction::SUM));

    return {result};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
