// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_translators.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reshape.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_bincount(const NodeContext& context) {
    num_inputs_check(context, 1, 3);
    auto input = context.get_input(0);

    Output<Node> weights;
    element::Type weights_type = element::f32;
    if (!context.input_is_none(1)) {
        weights = context.get_input(1);
        weights_type = weights.get_element_type();
        if (weights_type.is_dynamic()) {
            weights_type = element::f32;
        }
    } else {
        weights = context.mark_node(v0::Constant::create(element::i64, Shape{}, {1}));
        weights_type = element::i64;
    }

    Output<Node> minlength;
    if (!context.input_is_none(2)) {
        minlength = context.get_input(2);
    } else {
        minlength = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    }

    auto input_int = context.mark_node(std::make_shared<v0::Convert>(input, element::i32));
    auto minlength_i32 = context.mark_node(std::make_shared<v0::Convert>(minlength, element::i32));

    auto reduce_axes = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    auto max_val = context.mark_node(std::make_shared<v1::ReduceMax>(input_int, reduce_axes, true));

    auto max_length =
        context.mark_node(std::make_shared<v1::Add>(max_val, v0::Constant::create(element::i32, Shape{}, {1})));

    auto output_size = context.mark_node(std::make_shared<v1::Maximum>(max_length, minlength_i32));
    output_size = context.mark_node(
        std::make_shared<v1::Reshape>(output_size, v0::Constant::create(element::i32, Shape{1}, {1}), false));

    auto output_size_scalar = context.mark_node(
        std::make_shared<v1::Reshape>(output_size, v0::Constant::create(element::i32, Shape{0}, {}), false));

    auto result_vec = ov::frontend::common_translators::translate_bincount_common(context,
                                                                                  input_int,
                                                                                  output_size_scalar,
                                                                                  weights,
                                                                                  weights_type);

    return {result_vec[0]};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
