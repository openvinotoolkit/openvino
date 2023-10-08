// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include <vector>

namespace cldnn {

/// @brief generate proposals
struct generate_proposals
        : public primitive_base<generate_proposals> {
    CLDNN_DECLARE_PRIMITIVE(generate_proposals)

    generate_proposals() : primitive_base("", {}) {}

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
                       const std::vector<input_info>& inputs,
                       float min_size,
                       float nms_threshold,
                       int64_t pre_nms_count,
                       int64_t post_nms_count,
                       bool normalized,
                       float nms_eta,
                       const data_types roi_num_type,
                       const padding& output_padding = {}) :
            primitive_base{id, inputs, {output_padding}},
            output_rois_scores{inputs[4].pid},
            output_rois_num{inputs[5].pid},
            min_size{min_size},
            nms_threshold{nms_threshold},
            pre_nms_count{pre_nms_count},
            post_nms_count{post_nms_count},
            normalized{normalized},
            nms_eta{nms_eta},
            roi_num_type{roi_num_type} {}

    primitive_id output_rois_scores;
    primitive_id output_rois_num;
    float min_size = 0.0f;
    float nms_threshold = 0.0f;
    int64_t pre_nms_count = 0;
    int64_t post_nms_count = 0;
    bool normalized = false;
    float nms_eta = 0.0f;
    data_types roi_num_type = data_types::undefined;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, min_size);
        seed = hash_combine(seed, nms_threshold);
        seed = hash_combine(seed, pre_nms_count);
        seed = hash_combine(seed, post_nms_count);
        seed = hash_combine(seed, normalized);
        seed = hash_combine(seed, nms_eta);
        seed = hash_combine(seed, roi_num_type);
        seed = hash_combine(seed, output_rois_scores.empty());
        seed = hash_combine(seed, output_rois_num.empty());
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const generate_proposals>(rhs);

        #define cmp_fields(name) name == rhs_casted.name
        return cmp_fields(min_size) &&
               cmp_fields(nms_threshold) &&
               cmp_fields(pre_nms_count) &&
               cmp_fields(post_nms_count) &&
               cmp_fields(normalized) &&
               cmp_fields(nms_eta) &&
               cmp_fields(roi_num_type) &&
               cmp_fields(output_rois_scores.empty()) &&
               cmp_fields(output_rois_num.empty());
        #undef cmp_fields
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<generate_proposals>::save(ob);
        ob << output_rois_scores;
        ob << output_rois_num;
        ob << min_size;
        ob << nms_threshold;
        ob << pre_nms_count;
        ob << post_nms_count;
        ob << normalized;
        ob << nms_eta;
        ob << make_data(&roi_num_type, sizeof(data_types));
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<generate_proposals>::load(ib);
        ib >> output_rois_scores;
        ib >> output_rois_num;
        ib >> min_size;
        ib >> nms_threshold;
        ib >> pre_nms_count;
        ib >> post_nms_count;
        ib >> normalized;
        ib >> nms_eta;
        ib >> make_data(&roi_num_type, sizeof(data_types));
    }

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
}  // namespace cldnn
