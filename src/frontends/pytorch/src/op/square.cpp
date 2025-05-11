// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/power.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_square(const NodeContext& context) {
    // aten::square(Tensor self) -> Tensor
    num_inputs_check(context, 1, 1);
    auto input_0 = context.get_input(0);
    auto const_2 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {2}));
    const_2 = context.mark_node(std::make_shared<v1::ConvertLike>(const_2, input_0));
    return {context.mark_node(std::make_shared<v1::Power>(input_0, const_2))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov