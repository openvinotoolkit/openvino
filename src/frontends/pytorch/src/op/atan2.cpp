// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_translators.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

using namespace ov::op;

OutputVector translate_atan2(const NodeContext& context) {
    // atan2(input, other, *) → Tensor
    num_inputs_check(context, 2, 2);
    Output<Node> lhs;
    Output<Node> rhs;

    std::tie(lhs, rhs) = get_inputs_with_promoted_types(context, 0, 1);
    // torch.atan2 returns the default floating point type for integer and boolean inputs
    const auto dtype = lhs.get_element_type();
    if (dtype.is_static() && !dtype.is_real()) {
        lhs = context.mark_node(std::make_shared<v0::Convert>(lhs, element::f32));
        rhs = context.mark_node(std::make_shared<v0::Convert>(rhs, element::f32));
    }

    return common_translators::translate_atan2_util(context, lhs, rhs);
}

}  // namespace ov::frontend::pytorch::op
