// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_derive_index(const NodeContext& context) {
    // aten::__derive_index(int index, int start, int step) -> int
    num_inputs_check(context, 3, 3);
    auto index = context.get_input(0);
    auto start = context.get_input(1);
    auto step = context.get_input(2);
    auto index_step = context.mark_node(std::make_shared<v1::Multiply>(index, step));
    return {context.mark_node(std::make_shared<v1::Add>(start, index_step))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
