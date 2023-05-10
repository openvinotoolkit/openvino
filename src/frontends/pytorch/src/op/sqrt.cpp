// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/sqrt.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_sqrt(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto data = context.get_input(0);
    Output<Node> fake_const_for_type = context.mark_node(v0::Constant::create(element::f32, Shape({}), {.5}));
    align_eltwise_input_types(context, data, fake_const_for_type, true);
    return {context.mark_node(std::make_shared<v0::Sqrt>(data))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov