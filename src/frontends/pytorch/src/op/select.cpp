// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/gather.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_select(const NodeContext& context) {
    // aten::select.int(Tensor(a) self, int dim, SymInt index) -> Tensor(a)
    num_inputs_check(context, 3, 3);
    auto data = context.get_input(0);
    auto dim = context.get_input(1);
    auto index = context.get_input(2);
    return {context.mark_node(std::make_shared<v8::Gather>(data, index, dim))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
