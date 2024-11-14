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

    experimental_detectron_generate_proposals_single_image() : primitive_base("", {}) {}

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
           int64_t post_nms_count) :
            primitive_base{id, {input_im_info, input_anchors, input_deltas, input_scores, output_roi_scores}},
            output_roi_scores{output_roi_scores.pid},
            min_size{min_size},
            nms_threshold{nms_threshold},
            pre_nms_count{pre_nms_count},
            post_nms_count{post_nms_count} {}

    experimental_detectron_generate_proposals_single_image(const primitive_id& id,
           const input_info& input_im_info,
           const input_info& input_anchors,
           const input_info& input_deltas,
           const input_info& input_scores,
           float min_size,
           float nms_threshold,
           int64_t pre_nms_count,
           int64_t post_nms_count) :
            primitive_base{id, {input_im_info, input_anchors, input_deltas, input_scores}},
            output_roi_scores{},
            min_size{min_size},
            nms_threshold{nms_threshold},
            pre_nms_count{pre_nms_count},
            post_nms_count{post_nms_count} {}

    primitive_id output_roi_scores;
    float min_size = 0.0f;
    float nms_threshold = 0.0f;
    int64_t pre_nms_count = 0;
    int64_t post_nms_count = 0;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, min_size);
        seed = hash_combine(seed, nms_threshold);
        seed = hash_combine(seed, pre_nms_count);
        seed = hash_combine(seed, post_nms_count);
        seed = hash_combine(seed, output_roi_scores.empty());
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const experimental_detectron_generate_proposals_single_image>(rhs);

        return min_size == rhs_casted.min_size &&
               nms_threshold == rhs_casted.nms_threshold &&
               pre_nms_count == rhs_casted.pre_nms_count &&
               post_nms_count == rhs_casted.post_nms_count &&
               output_roi_scores.empty() == rhs_casted.output_roi_scores.empty();
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<experimental_detectron_generate_proposals_single_image>::save(ob);
        ob << output_roi_scores;
        ob << min_size;
        ob << nms_threshold;
        ob << pre_nms_count;
        ob << post_nms_count;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<experimental_detectron_generate_proposals_single_image>::load(ib);
        ib >> output_roi_scores;
        ib >> min_size;
        ib >> nms_threshold;
        ib >> pre_nms_count;
        ib >> post_nms_count;
    }

protected:
    std::vector<input_info> get_dependencies() const override {
        std::vector<input_info> ret;
        if (!output_roi_scores.empty())
            ret.push_back(output_roi_scores);
        return ret;
    }
};
}  // namespace cldnn
