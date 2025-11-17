// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/select.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_replace(const NodeContext& context) {
    num_inputs_check(context, 3, 3);
    auto input_tensor = context.get_input(0);
    auto target_value = context.get_input(1);
    auto replace_value = context.get_input(2);

    
    auto condition = std::make_shared<v1::Equal>(input_tensor, target_value);
    auto select_node = std::make_shared<v1::Select>(condition, replace_value, input_tensor);

    return {context.mark_node(select_node)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
