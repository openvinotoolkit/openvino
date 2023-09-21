// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <map>
#include <vector>

#include "ngraph/node.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/shape_util.hpp"
#include "nms_rotated_util.hpp"
#include "openvino/reference/non_max_suppression.hpp"

namespace ov {
namespace reference {
namespace nms_rotated {

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
                         const bool sort_result_descending,
                         const bool clockwise = true);

using ov::reference::nms_postprocessing;

}  // namespace nms_rotated

}  // namespace reference
}  // namespace ov
