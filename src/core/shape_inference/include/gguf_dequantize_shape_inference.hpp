// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/gguf_dequantize.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace internal {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const GGUFDequantize* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& = make_tensor_accessor()) {
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes.size() == 1);

    // [..., N, K / 256 * block_bytes] -> [..., N, K]
    auto output_shape = TRShape(input_shapes[0]);
    if (output_shape.rank().is_dynamic()) {
        return {std::move(output_shape)};
    }
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           output_shape.size() >= 2,
                           "The GGUF block data must have rank 2 or more ([..., N, bytes per row]). Got: ",
                           input_shapes[0]);

    auto& row = output_shape[output_shape.size() - 1];
    if (row.is_static()) {
        const auto bytes = static_cast<size_t>(row.get_length());
        const auto block_bytes = GGUFDequantize::block_bytes(op->get_format());
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               bytes % block_bytes == 0,
                               "Bytes per row (",
                               bytes,
                               ") must be a whole number of ",
                               op->get_format(),
                               " blocks of ",
                               block_bytes,
                               " bytes");
        row = static_cast<typename TRShape::value_type::value_type>(bytes / block_bytes * GGUFDequantize::block_size);
    } else {
        row = typename TRShape::value_type{};  // dynamic (an interval of bytes is not a whole number of blocks)
    }
    return {std::move(output_shape)};
}
}  // namespace internal
}  // namespace op
}  // namespace ov
