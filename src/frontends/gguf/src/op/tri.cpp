// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <memory>
#include <vector>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

// GGML_OP_TRI zeroes out elements outside a triangular region of a square matrix. The region is
// selected by the decoder-provided "tri_type" attribute (ggml_tri_type, mapped to a plain int):
//   0 = UPPER_DIAG : keep where col >= row
//   1 = UPPER      : keep where col >  row
//   2 = LOWER_DIAG : keep where col <= row
//   3 = LOWER      : keep where col <  row
//
// The last two OV axes are the square matrix; any leading axes are batch dimensions.
OutputVector translate_tri(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    auto x = context.get_input(0);

    auto tri_type = context.get_attribute<int>("tri_type");

    const auto shape = x.get_partial_shape();
    FRONT_END_OP_CONVERSION_CHECK(shape.rank().is_static() && shape.rank().get_length() >= 2,
                                  "TRI requires an operand of rank at least two");
    const auto rank = static_cast<size_t>(shape.rank().get_length());
    FRONT_END_OP_CONVERSION_CHECK(shape[rank - 2].is_static() && shape[rank - 1].is_static(),
                                  "TRI requires static matrix dimensions");
    const auto n = static_cast<size_t>(shape[rank - 1].get_length());
    FRONT_END_GENERAL_CHECK(shape[rank - 2].get_length() == static_cast<int64_t>(n),
                            "translate_tri: expected a square matrix, got ",
                            shape);

    std::vector<float> mask(n * n, 0.0f);
    for (size_t row = 0; row < n; ++row) {
        for (size_t col = 0; col < n; ++col) {
            bool keep = false;
            switch (tri_type) {
            case 0:
                keep = col >= row;
                break;  // UPPER_DIAG
            case 1:
                keep = col > row;
                break;  // UPPER
            case 2:
                keep = col <= row;
                break;  // LOWER_DIAG
            case 3:
                keep = col < row;
                break;  // LOWER
            default:
                FRONT_END_GENERAL_CHECK(false, "translate_tri: invalid tri_type ", tri_type);
            }
            mask[row * n + col] = keep ? 1.0f : 0.0f;
        }
    }
    // Build the mask in the node's own type, or an f16 input gets promoted to f32 by the Multiply.
    auto keep_mask = ov::op::v0::Constant::create(x.get_element_type(), ov::Shape{n, n}, mask);

    auto res = std::make_shared<ov::op::v1::Multiply>(x, keep_mask);

    return rename_outputs_with_suffix({std::move(res)}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
