// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <memory>
#include <vector>

#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"

#include "node_context.hpp"
#include "op_table.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

// GGML_OP_DIAG turns a vector into a diagonal matrix: a ggml [ne0, 1, ne2, ne3] input becomes
// [ne0, ne0, ne2, ne3] with output[..., i, j] = input[..., 0, j] when i == j, else 0.
// In OV layout (ggml [ne0, ne1, ne2, ne3] -> OV [ne3, ne2, ne1, ne0]) the input is
// [ne3, ne2, 1, ne0] and the output is [ne3, ne2, ne0, ne0]; the new row axis is OV axis 2.
//
// We multiply the input by a constant [1, 1, n, n] identity matrix: broadcasting spreads the input
// over the new row axis and the identity zeroes off-diagonal entries. This avoids an Equal/Select
// over a folded boolean constant, which the CPU Snippets body-hasher mishandles.
OutputVector translate_diag(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    auto x = context.get_input(0);  // OV shape: [ne3, ne2, 1, ne0]

    auto out_shape = context.get_output_shape().to_shape();
    size_t n = out_shape[3];  // ne0

    // Explicit [1, 1, n, n] identity matrix (1.0 on the diagonal, 0.0 elsewhere).
    std::vector<float> identity(n * n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        identity[i * n + i] = 1.0f;
    }
    auto eye = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, n, n}, identity);

    // Multiply broadcasts the [ne3,ne2,1,ne0] input over the row axis, zeroing off-diagonal entries.
    auto res = std::make_shared<ov::op::v1::Multiply>(x, eye);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
