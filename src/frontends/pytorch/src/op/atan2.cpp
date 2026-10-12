// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_translators.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

using namespace ov::op;

OutputVector translate_atan2(const NodeContext& context) {
    // atan2(input, other, *) → Tensor
    num_inputs_check(context, 2, 2);
    Output<Node> lhs;
    Output<Node> rhs;

    std::tie(lhs, rhs) = get_inputs_with_promoted_types(context, 0, 1);

    return common_translators::translate_atan2_util(context, lhs, rhs);
}

}  // namespace ov::frontend::pytorch::op
