// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_rsub(const NodeContext& context) {
    num_inputs_check(context, 2, 3);
    auto self = context.get_input(0);
    auto other = context.get_input(1);
    if (!context.input_is_none(2)) {
        auto alpha = context.get_input(2);
        align_eltwise_input_types(context, self, other);
        // reverse aten::sub other - self * alpha
        auto alpha_casted = context.mark_node(std::make_shared<v1::ConvertLike>(alpha, self));
        auto alpha_mul = context.mark_node(std::make_shared<v1::Multiply>(self, alpha_casted));
        return {context.mark_node(std::make_shared<v1::Subtract>(other, alpha_mul))};
    }
    align_eltwise_input_types(context, self, other);
    return {context.mark_node(std::make_shared<v1::Subtract>(other, self))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
