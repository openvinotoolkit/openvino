// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/sqrt.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_rsqrt(const NodeContext& context) {
    num_inputs_check(context, 1, 2);
    auto data = get_input_with_floating_type(context, 0);
    auto one_const = context.mark_node(v0::Constant::create(element::f32, Shape({}), {1}));
    auto one_const_casted = context.mark_node(std::make_shared<v1::ConvertLike>(one_const, data));
    auto sqrt_data = context.mark_node(std::make_shared<v0::Sqrt>(data));
    return {context.mark_node(std::make_shared<v1::Divide>(one_const_casted, sqrt_data))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
