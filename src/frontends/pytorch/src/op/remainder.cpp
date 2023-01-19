// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_remainder(NodeContext& context) {
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    auto div = context.mark_node(std::make_shared<ov::op::v1::Divide>(x, y, true));
    auto floor = context.mark_node(std::make_shared<ov::op::v0::Floor>(div));
    auto quo = context.mark_node(std::make_shared<ov::op::v1::Multiply>(floor, y));
    return {context.mark_node(std::make_shared<ov::op::v1::Subtract>(x, quo))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov