// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/exp.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_exp(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto data = context.get_input(0);
    // value of const doesn't really matter, we need it just for type alignment
    Output<Node> fake_e_const_for_type = context.mark_node(v0::Constant::create(element::f32, Shape({}), {2.71828}));
    align_eltwise_input_types(context, data, fake_e_const_for_type, true);
    return {context.mark_node(std::make_shared<v0::Exp>(data))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov