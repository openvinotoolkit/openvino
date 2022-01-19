// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
    /// @param sort_result_descending Specifies whether it is necessary to sort selected boxes across batches or not.
    /// @param num_select_per_class Id of primitive producing number of boxes to select per class.
    /// @param iou_threshold Id of primitive producing threshold value for IOU.
    /// @param score_threshold Id of primitive producing threshold value for scores.
    /// @param soft_nms_sigma Id of primitive specifying the sigma parameter for Soft-NMS.
    /// @param second_output Id of primitive specifying output for scores for each selected box.
    /// @param third_output Id of primitive specifying output for total number of selected boxes.
    non_max_suppression(const primitive_id& id,
                        const primitive_id& boxes_positions,
                        const primitive_id& boxes_score,
                        int selected_indices_num,
                        bool center_point_box = false,
                        bool sort_result_descending = true,
                        const primitive_id& num_select_per_class = primitive_id(),
                        const primitive_id& iou_threshold = primitive_id(),
                        const primitive_id& score_threshold = primitive_id(),
                        const primitive_id& soft_nms_sigma = primitive_id(),
                        const primitive_id& second_output = primitive_id(),
                        const primitive_id& third_output = primitive_id(),
                        const primitive_id& ext_prim_id = "")
        : primitive_base(id, {boxes_positions, boxes_score}, ext_prim_id)
        , selected_indices_num(selected_indices_num)
        , center_point_box(center_point_box)
        , sort_result_descending(sort_result_descending)
        , num_select_per_class(num_select_per_class)
        , iou_threshold(iou_threshold)
        , score_threshold(score_threshold)
        , soft_nms_sigma(soft_nms_sigma)
        , second_output(second_output)
        , third_output(third_output) {}

    int selected_indices_num;
    bool center_point_box;
    bool sort_result_descending;
    primitive_id num_select_per_class;
    primitive_id iou_threshold;
    primitive_id score_threshold;
    primitive_id soft_nms_sigma;
    primitive_id second_output;
    primitive_id third_output;

    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override {
        std::vector<std::reference_wrapper<const primitive_id>> ret;
        if (!num_select_per_class.empty())
            ret.push_back(num_select_per_class);
        if (!iou_threshold.empty())
            ret.push_back(iou_threshold);
        if (!score_threshold.empty())
            ret.push_back(score_threshold);
        if (!soft_nms_sigma.empty())
            ret.push_back(soft_nms_sigma);
        if (!second_output.empty())
            ret.push_back(second_output);
        if (!third_output.empty())
            ret.push_back(third_output);

        return ret;
    }
};
/// @}
/// @}
/// @}
}  // namespace cldnn
