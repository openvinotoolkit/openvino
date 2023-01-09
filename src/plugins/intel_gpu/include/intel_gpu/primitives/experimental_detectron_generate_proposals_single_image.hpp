// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include <vector>

namespace cldnn {

/// @brief experimental detectron generate proposals single image
struct experimental_detectron_generate_proposals_single_image
        : public primitive_base<experimental_detectron_generate_proposals_single_image> {
    CLDNN_DECLARE_PRIMITIVE(experimental_detectron_generate_proposals_single_image)

    /// @brief Constructs experimental_detectron_generate_proposals_single_image primitive
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
    experimental_detectron_generate_proposals_single_image(const primitive_id& id,
           const input_info& input_im_info,
           const input_info& input_anchors,
           const input_info& input_deltas,
           const input_info& input_scores,
           const input_info& output_roi_scores,
           float min_size,
           float nms_threshold,
           int64_t pre_nms_count,
           int64_t post_nms_count,
           const padding& output_padding = {}) :
            primitive_base{id, {input_im_info, input_anchors, input_deltas, input_scores, output_roi_scores}, {output_padding}},
            output_roi_scores{output_roi_scores.pid},
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
}  // namespace cldnn
