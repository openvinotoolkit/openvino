// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <memory>
#include <vector>

#include "openvino/frontend/exception.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"

#include "node_context.hpp"
#include "op_table.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

// GGML_OP_TRI zeroes out elements outside a triangular region of a square matrix. The region is
// selected by the decoder-provided "tri_type" attribute (ggml_tri_type, mapped to a plain int):
//   0 = UPPER_DIAG : keep where col >= row
//   1 = UPPER      : keep where col >  row
//   2 = LOWER_DIAG : keep where col <= row
//   3 = LOWER      : keep where col <  row
//
// In OV layout (ggml [ne0, ne1, ne2, ne3] -> OV [ne3, ne2, ne1, ne0]): ggml dim 0 (ne0, cols) is
// OV axis 3 and ggml dim 1 (ne1, rows) is OV axis 2. The matrix is square so ne0 == ne1.
//
// tri_type and n are both known at conversion time, so we build the exact [1,1,n,n] f32 keep-mask
// (1.0 to keep, 0.0 to zero) as a constant and multiply. This avoids a Range/Compare/Select chain
// over a folded boolean constant.
OutputVector translate_tri(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    auto x = context.get_input(0);  // OV shape: [ne3, ne2, ne1, ne0]

    auto tri_type = context.get_attribute<int>("tri_type");

    auto shape = context.get_input_shape(0).to_shape();
    size_t n = shape[3];  // ne0 == ne1

    std::vector<float> mask(n * n, 0.0f);
    for (size_t row = 0; row < n; ++row) {
        for (size_t col = 0; col < n; ++col) {
            bool keep = false;
            switch (tri_type) {
            case 0: keep = col >= row; break;  // UPPER_DIAG
            case 1: keep = col > row;  break;  // UPPER
            case 2: keep = col <= row; break;  // LOWER_DIAG
            case 3: keep = col < row;  break;  // LOWER
            default:
                FRONT_END_GENERAL_CHECK(false, "translate_tri: invalid tri_type ", tri_type);
            }
            mask[row * n + col] = keep ? 1.0f : 0.0f;
        }
    }
    auto keep_mask = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, n, n}, mask);

    auto res = std::make_shared<ov::op::v1::Multiply>(x, keep_mask);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
