// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/floor_mod.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_remainder(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    align_eltwise_input_types(context, x, y, true);
    return {context.mark_node(std::make_shared<v1::FloorMod>(x, y))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
