// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather_elements.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_gather(const NodeContext& context) {
    // aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor
    // aten::gather.out(Tensor self, int dim, Tensor index, *, bool sparse_grad=False, Tensor(a!) out) -> Tensor(a!)
    num_inputs_check(context, 3, 5);
    auto x = context.get_input(0);
    auto axis = context.const_input<int64_t>(1);
    auto index = context.get_input(2);
    index = context.mark_node(std::make_shared<ov::op::v0::Convert>(index, element::i32));
    // input 3 sparse_grad if True, gradient w.r.t. input will be a sparse tensor, used only for training, skip
    auto gather_elements = context.mark_node(std::make_shared<ov::op::v6::GatherElements>(x, index, axis));
    if (!context.input_is_none(4)) {
        context.mutate_input(4, gather_elements);
    }
    return {gather_elements};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
