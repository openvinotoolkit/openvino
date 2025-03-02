// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/is_nan.hpp"
#include "openvino/op/is_inf.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_nan_to_num_fx(const NodeContext& context) {
    num_inputs_check(context, 1, 4);
    
    auto x = context.get_input(0);
    
    Output<Node> nan_replacement = v0::Constant::create(element::f32, Shape{}, {0.0f});
    Output<Node> posinf_replacement = v0::Constant::create(element::f32, Shape{}, {std::numeric_limits<float>::max()});
    Output<Node> neginf_replacement = v0::Constant::create(element::f32, Shape{}, {std::numeric_limits<float>::lowest()});

    if (!context.input_is_none(1)) {
        nan_replacement = context.get_input(1);
    }
    if (!context.input_is_none(2)) {
        posinf_replacement = context.get_input(2);
    }
    if (!context.input_is_none(3)) {
        neginf_replacement = context.get_input(3);
    }

    auto is_nan = context.mark_node(std::make_shared<v1::Equal>(x, v0::Constant::create(element::f32, Shape{}, {std::nanf("")})));
    auto is_posinf = context.mark_node(std::make_shared<v1::Equal>(x, v0::Constant::create(element::f32, Shape{}, {std::numeric_limits<float>::infinity()})));
    auto is_neginf = context.mark_node(std::make_shared<v1::Equal>(x, v0::Constant::create(element::f32, Shape{}, {-std::numeric_limits<float>::infinity()})));

    auto replaced_nan = context.mark_node(std::make_shared<v1::Select>(is_nan, nan_replacement, x));
    auto replaced_posinf = context.mark_node(std::make_shared<v1::Select>(is_posinf, posinf_replacement, replaced_nan));
    auto replaced_neginf = context.mark_node(std::make_shared<v1::Select>(is_neginf, neginf_replacement, replaced_posinf));

    return {replaced_neginf};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
