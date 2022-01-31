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

/// @brief experimental detectron generate proposals single image
struct edgpsi
        : public primitive_base<edgpsi> {
    CLDNN_DECLARE_PRIMITIVE(edgpsi)

    /// @brief Constructs edgpsi primitive
    /// @param id This primitive id
    /// @param input_im_info image size info
    /// @param input_anchors anchors
    /// @param input_deltas deltas for anchors
    /// @param input_scores proposal scores
    /// @param output_roi_scores ROI scores
    /// @param min_size  minimum box width and height
    /// @param nms_threshold threshold to be used in NonMaxSuppression stage
    /// @param pre_nms_count number of top-n proposals before NMS
    /// @param post_nms_count number of top-n proposals after NMS
    edgpsi(const primitive_id& id,
           const primitive_id& input_im_info,
           const primitive_id& input_anchors,
           const primitive_id& input_deltas,
           const primitive_id& input_scores,
           const primitive_id& output_roi_scores,
           float min_size,
           float nms_threshold,
           int64_t pre_nms_count,
           int64_t post_nms_count,
           const primitive_id& ext_prim_id = "",
           const padding& output_padding = {}) :
            primitive_base{id, {input_im_info, input_anchors, input_deltas, input_scores, output_roi_scores}, ext_prim_id, output_padding},
            output_roi_scores{output_roi_scores},
            min_size{min_size},
            nms_threshold{nms_threshold},
            pre_nms_count{pre_nms_count},
            post_nms_count{post_nms_count} {}

    primitive_id output_roi_scores;
    float min_size;
    float nms_threshold;
    int64_t pre_nms_count;
    int64_t post_nms_count;

protected:
    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override {
        std::vector<std::reference_wrapper<const primitive_id>> ret;
        if (!output_roi_scores.empty())
            ret.push_back(output_roi_scores);
        return ret;
    }
};
/// @}
/// @}
/// @}
}  // namespace cldnn
