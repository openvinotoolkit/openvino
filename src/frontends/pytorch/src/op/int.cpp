// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

OutputVector translate_int(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    return {context.mark_node(std::make_shared<ov::op::v0::Convert>(context.get_input(0), element::i64))};
};

}  // namespace ov::frontend::pytorch::op
