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

OutputVector translate_sub_common(const NodeContext& context, bool inplace) {
    num_inputs_check(context, 2, 3);
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    if (inplace) {
        if (x.get_element_type().is_dynamic() || x.get_element_type() != y.get_element_type())
            y = context.mark_node(std::make_shared<v1::ConvertLike>(y, x));
    } else {
        align_eltwise_input_types(context, x, y);
    }
    // default alpha is 1 so no need to multiply if alpha is not provided
    if (!context.input_is_none(2)) {
        auto alpha = context.get_input(2);
        auto casted_alpha = context.mark_node(std::make_shared<v1::ConvertLike>(alpha, y));
        y = context.mark_node(std::make_shared<v1::Multiply>(casted_alpha, y));
    }
    auto sub = context.mark_node(std::make_shared<v1::Subtract>(x, y));
    if (inplace)
        context.mutate_input(0, sub);
    return {sub};
};

OutputVector translate_sub(const NodeContext& context) {
    return translate_sub_common(context, false);
};

OutputVector translate_sub_(const NodeContext& context) {
    return translate_sub_common(context, true);
};

OutputVector translate_sub_fx(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    align_eltwise_input_types(context, x, y);
    // default alpha is 1 so no need to multiply if alpha is not provided
    if (context.has_attribute("alpha")) {
        auto alpha = context.get_attribute<Output<Node>>("alpha");
        auto casted_alpha = context.mark_node(std::make_shared<v1::ConvertLike>(alpha, y));
        y = context.mark_node(std::make_shared<v1::Multiply>(casted_alpha, y));
    }
    return {context.mark_node(std::make_shared<v1::Subtract>(x, y))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov