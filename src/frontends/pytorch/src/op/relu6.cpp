// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/clamp.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

OutputVector translate_relu6(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto x = context.get_input(0);
    return {context.mark_node(std::make_shared<ov::op::v0::Clamp>(x, 0., 6.))};
};

}  // namespace ov::frontend::pytorch::op