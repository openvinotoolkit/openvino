// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace reference {
void non_max_suppression5(const float* boxes_data,
                          const Shape& boxes_data_shape,
                          const float* scores_data,
                          const Shape& scores_data_shape,
                          int64_t max_output_boxes_per_class,
                          float iou_threshold,
                          float score_threshold,
                          float soft_nms_sigma,
                          int64_t* selected_indices,
                          const Shape& selected_indices_shape,
                          float* selected_scores,
                          const Shape& selected_scores_shape,
                          int64_t* valid_outputs,
                          const bool sort_result_descending);

void nms5_postprocessing(ov::TensorVector& outputs,
                         const element::Type output_type,
                         const std::vector<int64_t>& selected_indices,
                         const std::vector<float>& selected_scores,
                         int64_t valid_outputs,
                         const element::Type selected_scores_type);

void non_max_suppression(const float* boxes_data,
                         const Shape& boxes_data_shape,
                         const float* scores_data,
                         const Shape& scores_data_shape,
                         int64_t max_output_boxes_per_class,
                         float iou_threshold,
                         float score_threshold,
                         float soft_nms_sigma,
                         int64_t* selected_indices,
                         const Shape& selected_indices_shape,
                         float* selected_scores,
                         const Shape& selected_scores_shape,
                         int64_t* valid_outputs,
                         const bool sort_result_descending);

void nms_postprocessing(ov::TensorVector& outputs,
                        const element::Type output_type,
                        const std::vector<int64_t>& selected_indices,
                        const std::vector<float>& selected_scores,
                        int64_t valid_outputs,
                        const element::Type selected_scores_type);
}  // namespace reference
}  // namespace ov
