// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/squeeze.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_squeeze(const NodeContext& context) {
    num_inputs_check(context, 1, 2);
    auto x = context.get_input(0);
    if (context.input_is_none(1)) {
        return {context.mark_node(std::make_shared<v0::Squeeze>(x))};
    }
    return {context.mark_node(std::make_shared<v0::Squeeze>(x, context.get_input(1)))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
