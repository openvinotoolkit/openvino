// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/frontend/sparse_type_mark.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_sparse_mm(const NodeContext& context) {
    // aten::_sparse_mm(Tensor sparse, Tensor matrix) -> Tensor
    num_inputs_check(context, 2, 2);
    auto sparse_mat = context.get_input(0);
    auto mat2 = context.get_input(1);

    // Delegate to SparseTypeMark aware implementation
    return {ov::frontend::SparseTypeMark::sparse_mm(context, sparse_mat, mat2)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
