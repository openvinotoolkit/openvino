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
        auto input = context.get_input_from_visible_context(i);
        if (!ov::as_type_ptr<v0::Constant>(input.get_node_shared_ptr())) {
            input = context.get_input(static_cast<int>(i));
        }
        inputs.push_back(input);
    }
    auto list_construct = context.mark_node(make_list_construct(inputs));
    return {list_construct};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
