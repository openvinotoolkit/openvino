// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/prelu.hpp"
#include "utils.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/add.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_rrelu(const NodeContext& context) {
    num_inputs_check(context, 1, 3);
    auto x = context.get_input(0);

    float default_negative_slope = 0.2291666666666667f
    float default_lower = 0.125f;
    float default_upper = 0.3333333333333333f;

    Output<Node> negative_slope = ov::op::v0::Constant::create(element::f32, Shape{1}, {default_negative_slope});
    if (context.get_input_size() == 1){
        negative_slope = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(negative_slope, x));
    }
    else{
        Output<Node> lower = ov::op::v0::Constant::create(element::f32, Shape{1}, {default_lower});
        Output<Node> upper = ov::op::v0::Constant::create(element::f32, Shape{1}, {default_upper});
        if(!context.input_is_none(1)){
            lower = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(context.get_input(1), x));
        }
        else{
            lower = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(lower, x));
        }
        if(!context.input_is_none(2)){
            upper = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(context.get_input(2), x));
        }
        else{
            upper = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(upper, x));
        }

        negative_slope = context.mark_node(std::make_shared<ov::op::v1::Divide>(
            std::make_shared<ov::op::v1::Add>(lower, upper),
            ov::op::v0::Constant::create(x.get_element_type(), Shape{1}, {2.0})
        ));
    }
    return {context.mark_node(std::make_shared<v0::PRelu>(x, negative_slope))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
