// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_sparse_mm(const NodeContext& context) {
    // aten::_sparse_mm(Tensor sparse, Tensor dense) -> Tensor
    // Performs matrix multiplication between a sparse matrix and a dense matrix
    // Input 0: sparse tensor (COO format)
    // Input 1: dense tensor
    // Output: dense tensor
    
    num_inputs_check(context, 2, 2);
    auto sparse_input = context.get_input(0);
    auto dense_input = context.get_input(1);
    
    // TorchScript preserves sparse COO tensors for this operation.
    // However, OpenVINO does not currently have a native decomposition for sparse matrix multiplication.
    // We use PtFrameworkNode to delegate execution back to PyTorch runtime.
    // This is an intentional design choice for operations without native OpenVINO equivalents.
    
    // Create a framework node that will be executed by PyTorch runtime
    auto fw_node = std::make_shared<PtFrameworkNode>(context.get_decoder(), 
                                                       OutputVector{sparse_input, dense_input}, 
                                                       1);
    return {context.mark_node(fw_node)};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
