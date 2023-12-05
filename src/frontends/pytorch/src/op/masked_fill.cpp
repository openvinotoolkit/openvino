// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_masked_fill(const NodeContext& context) {
    num_inputs_check(context, 3, 3);
    auto data = context.get_input(0);
    auto mask = context.get_input(1);
    auto value = context.get_input(2);
    value = context.mark_node(std::make_shared<v1::ConvertLike>(value, data));
    auto bool_mask = context.mark_node(std::make_shared<v0::Convert>(mask, element::boolean));
    return {context.mark_node(std::make_shared<v1::Select>(bool_mask, value, data))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
