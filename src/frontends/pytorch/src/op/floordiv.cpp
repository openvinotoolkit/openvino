// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/divide.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_floordiv(NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    return {context.mark_node(std::make_shared<ov::op::v1::Divide>(x, y, true))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov