// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/op/extractimagepatches.hpp>

namespace ov {
namespace op {
namespace v3 {
template <class T>
void shape_infer(const ExtractImagePatches* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1 && output_shapes.size() == 1);
    const auto& input_shape = input_shapes[0];
    auto& output_shape = output_shapes[0];
    output_shape.resize(4);
    NODE_VALIDATION_CHECK(op, input_shape.rank() == 4, "input tensor must be 4D tensor.");

    NODE_VALIDATION_CHECK(op,
                          op->m_patch_sizes.size() == 2,
                          "Attribute sizes should be in [size_rows, size_cols] format.");

    NODE_VALIDATION_CHECK(op,
                          op->m_patch_movement_strides.size() == 2,
                          "Attribute strides should be in [stride_rows, stride_cols] format.");

    NODE_VALIDATION_CHECK(op,
                          op->m_patch_movement_strides[0] > 0 && op->m_patch_movement_strides[1] > 0,
                          "Attribute strides should be strictly greater than zeros in values.");

    NODE_VALIDATION_CHECK(op,
                          op->m_patch_selection_rates.size() == 2,
                          "Attribute rates should be in [rate_rows, rate_cols] format.");

    NODE_VALIDATION_CHECK(op,
                          op->m_patch_selection_rates[0] > 0 && op->m_patch_selection_rates[1] > 0,
                          "Attribute rates should be strictly greater than zeros in values.");

    NODE_VALIDATION_CHECK(
        op,
        op->m_padding == PadType::VALID || op->m_padding == PadType::SAME_LOWER || op->m_padding == PadType::SAME_UPPER,
        "Attribute padding should be in either valid or same_lower or same_upper.");

    if (input_shape[1].is_dynamic() || input_shape[2].is_dynamic() || input_shape[3].is_dynamic()) {
        return;
    } else {
        int32_t input_depth = input_shape[1].get_length();
        int32_t input_rows = input_shape[2].get_length();
        int32_t input_cols = input_shape[3].get_length();
        int32_t out_rows(0);
        int32_t out_cols(0);

        if (input_rows == 0 || input_cols == 0) {
            out_rows = 0;
            out_cols = 0;
        } else if (op->m_padding == PadType::VALID) {
            out_rows =
                (((input_rows) -
                  static_cast<int32_t>(op->m_patch_selection_rates[0]) * (static_cast<int32_t>(op->m_patch_sizes[0]) - 1) - 1) /
                 op->m_patch_movement_strides[0]) +
                1;
            out_cols =
                (((input_cols) -
                  static_cast<int32_t>(op->m_patch_selection_rates[1]) * (static_cast<int32_t>(op->m_patch_sizes[1]) - 1) - 1) /
                 op->m_patch_movement_strides[1]) +
                1;
        } else {
            out_rows = 1 + (((input_rows)-1) / op->m_patch_movement_strides[0]);
            out_cols = 1 + (((input_cols)-1) / op->m_patch_movement_strides[1]);
        }

        if (out_rows < 0)
            out_rows = 0;
        if (out_cols < 0)
            out_cols = 0;

        auto out_depth_cast =
            static_cast<ngraph::Dimension::value_type>(input_depth * op->m_patch_sizes[0] * op->m_patch_sizes[1]);
        auto out_rows_cast = static_cast<ngraph::Dimension::value_type>(out_rows);
        auto out_cols_cast = static_cast<ngraph::Dimension::value_type>(out_cols);

        if (input_shape[0].is_dynamic()) {
            output_shape[0] = input_shape[0];
            output_shape[1] = out_depth_cast;
            output_shape[2] = out_rows_cast;
            output_shape[3] = out_cols_cast;
        } else {
            output_shape[0] = input_shape[0].get_length();
            output_shape[1] = out_depth_cast;
            output_shape[2] = out_rows_cast;
            output_shape[3] = out_cols_cast;
        }

        if (input_rows == 0 || input_cols == 0) {
            output_shape = input_shape;
        }
    }
}
}  // namespace v3
}  // namespace op
}  // namespace ov
