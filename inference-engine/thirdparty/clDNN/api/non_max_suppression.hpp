/*
// Copyright (c) 2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "primitive.hpp"

#include <vector>

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Performs non max supression of input boxes and returns indices of selected boxes.
/// @detail Filters out boxes that have high intersection-over-union (IOU) with previously
/// selected boxes with higher score. Boxes with score higher than score_threshold are
/// filtered out. This filtering happens per class.
struct non_max_suppression : public primitive_base<non_max_suppression> {
    CLDNN_DECLARE_PRIMITIVE(non_max_suppression)

    /// @brief Creates non max supression primitive.
    /// @param id This primitive id.
    /// @param boxes_positions Id of primitive with bounding boxes.
    /// @param boxes_score Id of primitive with boxes scores per class.
    /// @param selected_indices_num Number of selected indices.
    /// @param center_point_box If true boxes are represented as [center x, center y, width, height].
    /// @param num_select_per_class Id of primitive producing number of boxes to select per class.
    /// @param iou_threshold Id of primitive producing threshold value for IOU.
    /// @param score_threshold Id of primitive producing threshold value for scores.
    non_max_suppression(const primitive_id& id,
                       const primitive_id& boxes_positions,
                       const primitive_id& boxes_score,
                       int selected_indices_num,
                       bool center_point_box = false,
                       const primitive_id& num_select_per_class = primitive_id(),
                       const primitive_id& iou_threshold = primitive_id(),
                       const primitive_id& score_threshold = primitive_id())
        : primitive_base(id, {boxes_positions, boxes_score})
        , selected_indices_num(selected_indices_num)
        , center_point_box(center_point_box)
        , num_select_per_class(num_select_per_class)
        , iou_threshold(iou_threshold)
        , score_threshold(score_threshold) {}

    int selected_indices_num;
    bool center_point_box;
    primitive_id num_select_per_class;
    primitive_id iou_threshold;
    primitive_id score_threshold;

    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override {
        std::vector<std::reference_wrapper<const primitive_id>> ret;
        if (!num_select_per_class.empty())
            ret.push_back(num_select_per_class);
        if (!iou_threshold.empty())
            ret.push_back(iou_threshold);
        if (!score_threshold.empty())
            ret.push_back(score_threshold);
        return ret;
    }
};
/// @}
/// @}
/// @}
}  // namespace cldnn
