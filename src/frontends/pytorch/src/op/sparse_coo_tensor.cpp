// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/frontend/sparse_type_mark.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_sparse_coo_tensor(const NodeContext& context) {
    // aten::sparse_coo_tensor(indices, values, size, dtype=None, layout=None, device=None, pin_memory=False)
    num_inputs_check(context, 3, 7);
    auto indices = context.get_input(0);  // [ndim, nnz]
    auto values = context.get_input(1);   // [nnz]
    auto size = context.get_input(2);     // [ndim] - shape of dense tensor

    // Defer dense conversion via SparseTypeMark
    auto sparse_mark = context.mark_node(
        std::make_shared<ov::frontend::SparseTypeMark>(indices, values, size, values.get_element_type()));

    return {sparse_mark};
};

OutputVector translate_to_dense(const NodeContext& context) {
    // aten::to_dense(Tensor self, ScalarType? dtype=None, bool? masked_grad=None) -> Tensor
    num_inputs_check(context, 1, 3);
    auto input = context.get_input(0);
    auto sparse_mark = std::dynamic_pointer_cast<ov::frontend::SparseTypeMark>(input.get_node_shared_ptr());
    if (sparse_mark) {
        return {sparse_mark->to_dense(context)};
    }
    // Already dense, return as-is
    return {input};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
