// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/op/extractimagepatches.hpp>

#include "utils.hpp"

namespace ov {
namespace op {
namespace v3 {
int32_t inline calc_shape_padding(const int32_t input,
                                  const int32_t rate,
                                  const int32_t stride,
                                  const int32_t patch_size,
                                  const PadType type) {
    int32_t out = 0;
    if (type == PadType::VALID) {
        out = (input - rate * (patch_size - 1) - 1) / stride + 1;
    } else {
        out = 1 + (input - 1) / stride;
    }
    return out < 0 ? 0 : out;
}

template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const ExtractImagePatches* op, const std::vector<T>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1);
    using DimType = typename T::value_type;
    const auto& input_shape = input_shapes[0];
    auto output_shapes = std::vector<TRShape>(1);
    auto& output_shape = output_shapes[0];
    output_shape.resize(4);
    NODE_VALIDATION_CHECK(op, input_shape.rank().compatible(4), "input tensor must be 4D tensor.");

    NODE_VALIDATION_CHECK(op,
                          op->get_sizes().size() == 2,
                          "Attribute sizes should be in [size_rows, size_cols] format.");

    NODE_VALIDATION_CHECK(op,
                          op->get_strides().size() == 2,
                          "Attribute strides should be in [stride_rows, stride_cols] format.");

    NODE_VALIDATION_CHECK(op,
                          op->get_strides()[0] > 0 && op->get_strides()[1] > 0,
                          "Attribute strides should be strictly greater than zeros in values.");

    NODE_VALIDATION_CHECK(op,
                          op->get_rates().size() == 2,
                          "Attribute rates should be in [rate_rows, rate_cols] format.");

    NODE_VALIDATION_CHECK(op,
                          op->get_rates()[0] > 0 && op->get_rates()[1] > 0,
                          "Attribute rates should be strictly greater than zeros in values.");

    NODE_VALIDATION_CHECK(op,
                          op->get_auto_pad() == PadType::VALID || op->get_auto_pad() == PadType::SAME_LOWER ||
                              op->get_auto_pad() == PadType::SAME_UPPER,
                          "Attribute padding should be in either valid or same_lower or same_upper.");

    if (input_shape.rank().is_static()) {
        // Determine batch & output_depth
        output_shape[0] = input_shape[0];
        output_shape[1] = input_shape[1] * op->get_sizes()[0] * op->get_sizes()[1];
        // Determine spatial shape
        if (input_shape[2].is_static() && input_shape[3].is_static()) {
            int32_t input_rows = static_cast<int32_t>(input_shape[2].get_length());
            int32_t input_cols = static_cast<int32_t>(input_shape[3].get_length());
            int32_t out_rows(0);
            int32_t out_cols(0);
            if (input_rows == 0 || input_cols == 0) {
                output_shape = input_shape;
                return output_shapes;
            }

            out_rows = calc_shape_padding(input_rows,
                                          static_cast<int32_t>(op->get_rates()[0]),
                                          static_cast<int32_t>(op->get_strides()[0]),
                                          static_cast<int32_t>(op->get_sizes()[0]),
                                          op->get_auto_pad());
            out_cols = calc_shape_padding(input_cols,
                                          static_cast<int32_t>(op->get_rates()[1]),
                                          static_cast<int32_t>(op->get_strides()[1]),
                                          static_cast<int32_t>(op->get_sizes()[1]),
                                          op->get_auto_pad());

            auto out_rows_cast = static_cast<typename DimType::value_type>(out_rows);
            auto out_cols_cast = static_cast<typename DimType::value_type>(out_cols);

            output_shape[2] = out_rows_cast;
            output_shape[3] = out_cols_cast;
        }
    }
    return output_shapes;
}
}  // namespace v3
}  // namespace op
}  // namespace ov
