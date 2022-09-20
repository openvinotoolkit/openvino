// Copyright (C) 2022 Intel Corporation
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

/// @brief generate proposals
struct generate_proposals
        : public primitive_base<generate_proposals> {
    CLDNN_DECLARE_PRIMITIVE(generate_proposals)

    /// @brief Constructs generate_proposals primitive
    /// @param id This primitive id
    /// @param input_im_info image size info
    /// @param input_anchors anchors
    /// @param input_deltas deltas for anchors
    /// @param input_scores proposal scores
    /// @param output_rois_scores ROIs scores
    /// @param output_rois_num number of proposed ROIs
    /// @param min_size  minimum box width and height
    /// @param nms_threshold threshold to be used in NonMaxSuppression stage
    /// @param pre_nms_count number of top-n proposals before NMS
    /// @param post_nms_count number of top-n proposals after NMS
    /// @param normalized indicates whether proposal bboxes are normalized
    /// @param nms_eta eta parameter for adaptive NMS
    /// @param roi_num_type type of 3rd output elements
    generate_proposals(const primitive_id& id,
                       const std::vector<primitive_id>& inputs,
                       float min_size,
                       float nms_threshold,
                       int64_t pre_nms_count,
                       int64_t post_nms_count,
                       bool normalized,
                       float nms_eta,
                       const data_types roi_num_type,
                       const padding& output_padding = {}) :
            primitive_base{id, inputs, output_padding},
            output_rois_scores{inputs[4]},
            output_rois_num{inputs[5]},
            min_size{min_size},
            nms_threshold{nms_threshold},
            pre_nms_count{pre_nms_count},
            post_nms_count{post_nms_count},
            normalized{normalized},
            nms_eta{nms_eta},
            roi_num_type{roi_num_type} {}

    primitive_id output_rois_scores;
    primitive_id output_rois_num;
    float min_size;
    float nms_threshold;
    int64_t pre_nms_count;
    int64_t post_nms_count;
    bool normalized;
    float nms_eta;
    data_types roi_num_type;

protected:
    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override {
        std::vector<std::reference_wrapper<const primitive_id>> ret;
        if (!output_rois_scores.empty())
            ret.push_back(output_rois_scores);
        if (!output_rois_num.empty())
            ret.push_back(output_rois_num);
        return ret;
    }
};
/// @}
/// @}
/// @}
}  // namespace cldnn
