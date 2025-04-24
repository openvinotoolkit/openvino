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

OutputVector translate_index_select(const NodeContext& context) {
    // aten::index_select(Tensor self, int dim, Tensor index) -> Tensor
    // aten::index_select.out(Tensor self, int dim, Tensor index, *, Tensor(a!) out) -> Tensor(a!)
    num_inputs_check(context, 3, 4);
    auto x = context.get_input(0);
    auto dim = context.get_input(1);
    auto indicies = context.get_input(2);
    auto gather = context.mark_node(std::make_shared<ov::op::v8::Gather>(x, indicies, dim));
    if (!context.input_is_none(3)) {
        context.mutate_input(3, gather);
    }
    return {gather};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
