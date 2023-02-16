// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_remainder(NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    auto div = context.mark_node(std::make_shared<v1::Divide>(x, y, true));
    auto floor = context.mark_node(std::make_shared<v0::Floor>(div));
    auto quo = context.mark_node(std::make_shared<v1::Multiply>(floor, y));
    return {context.mark_node(std::make_shared<v1::Subtract>(x, quo))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov