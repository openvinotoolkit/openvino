// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/sign.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

using namespace ov::op;

OutputVector translate_sign(const NodeContext& context) {
    // aten::sign(input, *, out=None)
    num_inputs_check(context, 1, 2);
    auto input = context.get_input(0);
    auto sign = context.mark_node(std::make_shared<ov::op::v0::Sign>(input));
    if (!context.input_is_none(1)) {
        context.mutate_input(1, sign);
    }
    return {sign};
};

}  // namespace ov::frontend::pytorch::op