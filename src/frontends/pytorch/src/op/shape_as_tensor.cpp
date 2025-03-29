// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_shape_as_tensor(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto input = context.get_input(0);
    auto shape = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i64));
    return {shape};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
