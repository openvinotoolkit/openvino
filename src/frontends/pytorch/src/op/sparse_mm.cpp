// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/matmul.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_sparse_mm(const NodeContext& context) {
    // aten::_sparse_mm(Tensor sparse, Tensor dense) -> Tensor
    num_inputs_check(context, 2, 2);
    auto sparse_mat = context.get_input(0);
    auto dense_mat = context.get_input(1);

    // We map _sparse_mm to a standard MatMul.
    // OpenVINO core/plugins will handle the execution. 
    // If the input is sparse-encoded, specific transformations might be needed, 
    // but at the frontend level, we represent it as a MatMul operation.
    return {context.mark_node(std::make_shared<ov::op::v0::MatMul>(sparse_mat, dense_mat, false, false))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
