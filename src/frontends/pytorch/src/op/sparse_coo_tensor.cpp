// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/transpose.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_sparse_coo_tensor(const NodeContext& context) {
    // aten::sparse_coo_tensor(indices, values, size, dtype=None, layout=None, device=None, pin_memory=False)
    num_inputs_check(context, 3, 7);
    auto indices = context.get_input(0);
    auto values = context.get_input(1);
    auto size = context.get_input(2);

    // indices shape is [ndim, nnz]
    // values shape is [nnz]
    // size is the shape of the resulting tensor

    // Create a zero-filled dense tensor of 'size' with type of 'values'
    auto zero_const = context.mark_node(v0::Constant::create(values.get_element_type(), Shape{}, {0}));
    auto dense_zero = context.mark_node(std::make_shared<v3::Broadcast>(zero_const, size));

    // ScatterNDUpdate expects indices as [nnz, ndim]
    // We need to transpose indices from [ndim, nnz] to [nnz, ndim]
    auto perm_order = context.mark_node(v0::Constant::create(element::i32, Shape{2}, {1, 0}));
    auto permuted_indices = context.mark_node(std::make_shared<v1::Transpose>(indices, perm_order));

    // Perform ScatterNDUpdate
    auto dense_tensor = context.mark_node(std::make_shared<v3::ScatterNDUpdate>(dense_zero, permuted_indices, values));

    return {dense_tensor};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
