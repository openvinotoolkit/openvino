// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/tile.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_repeat_fx(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto repeats = get_input_concat_if_list(context, 1);
    return {context.mark_node(std::make_shared<v0::Tile>(context.get_input(0), repeats))};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
