// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sqrt.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_rsqrt(NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto data = context.get_input(0);
    auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(data));
    auto one_const = context.mark_node(v0::Constant::create(element::f32, Shape({}), {1}));
    auto sqrt_data = context.mark_node(std::make_shared<v0::Sqrt>(data));
    return {context.mark_node(std::make_shared<v1::Divide>(one_const, sqrt_data))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov