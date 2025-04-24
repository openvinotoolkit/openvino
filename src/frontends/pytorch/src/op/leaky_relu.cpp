// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/prelu.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_leaky_relu_fx(const NodeContext& context) {
    num_inputs_check(context, 1, 2);
    auto x = context.get_input(0);
    float default_negative_slope = 1e-2f;
    Output<Node> negative_slope = ov::op::v0::Constant::create(element::f32, Shape{1}, {default_negative_slope});
    if (context.get_input_size() == 1) {
        negative_slope = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(negative_slope, x));
    } else {
        negative_slope = context.get_input(1);
    }
    return {context.mark_node(std::make_shared<v0::PRelu>(x, negative_slope))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
