// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/shape.hpp"
#include "openvino/reference/non_max_suppression.hpp"

namespace ov {
namespace reference {

void nms_rotated(const float* boxes_data,
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
                 bool sort_result_descending,
                 bool clockwise = true);

constexpr auto nms_rotated_postprocessing = ov::reference::nms_postprocessing;

}  // namespace reference
}  // namespace ov
