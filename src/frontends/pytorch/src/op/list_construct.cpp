// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/frontend/sequence_mark.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_list_construct(const NodeContext& context) {
    ov::OutputVector inputs;
    for (size_t i = 0; i < context.get_input_size(); i++) {
        inputs.push_back(context.get_input_from_visible_context(i));
    }
    auto list_construct = context.mark_node(make_list_construct(inputs));
    return {list_construct};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
