// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "openvino/op/generate_proposals.hpp"
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
                       const data_types roi_num_type) :
            primitive_base{id, inputs},
            output_rois_scores{inputs[4].pid},
            output_rois_num{inputs[5].pid},
            roi_num_type{roi_num_type} {
        attrs.min_size = min_size;
        attrs.nms_threshold = nms_threshold;
        attrs.pre_nms_count = pre_nms_count;
        attrs.post_nms_count = post_nms_count;
        attrs.normalized = normalized;
        attrs.nms_eta = nms_eta;
    }

    generate_proposals(const primitive_id& id,
                       const std::vector<input_info>& inputs,
                       const ov::op::v9::GenerateProposals::Attributes& attrs) :
            primitive_base{id, inputs, {}},
            attrs{attrs} {}

    ov::op::v9::GenerateProposals::Attributes attrs;

    primitive_id output_rois_scores;
    primitive_id output_rois_num;
    data_types roi_num_type = data_types::dynamic;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, attrs.min_size);
        seed = hash_combine(seed, attrs.nms_threshold);
        seed = hash_combine(seed, attrs.pre_nms_count);
        seed = hash_combine(seed, attrs.post_nms_count);
        seed = hash_combine(seed, attrs.normalized);
        seed = hash_combine(seed, attrs.nms_eta);
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
        return cmp_fields(attrs.min_size) &&
               cmp_fields(attrs.nms_threshold) &&
               cmp_fields(attrs.pre_nms_count) &&
               cmp_fields(attrs.post_nms_count) &&
               cmp_fields(attrs.normalized) &&
               cmp_fields(attrs.nms_eta) &&
               cmp_fields(roi_num_type) &&
               cmp_fields(output_rois_scores.empty()) &&
               cmp_fields(output_rois_num.empty());
        #undef cmp_fields
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<generate_proposals>::save(ob);
        ob << output_rois_scores;
        ob << output_rois_num;
        ob << attrs.min_size;
        ob << attrs.nms_threshold;
        ob << attrs.pre_nms_count;
        ob << attrs.post_nms_count;
        ob << attrs.normalized;
        ob << attrs.nms_eta;
        ob << make_data(&roi_num_type, sizeof(data_types));
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<generate_proposals>::load(ib);
        ib >> output_rois_scores;
        ib >> output_rois_num;
        ib >> attrs.min_size;
        ib >> attrs.nms_threshold;
        ib >> attrs.pre_nms_count;
        ib >> attrs.post_nms_count;
        ib >> attrs.normalized;
        ib >> attrs.nms_eta;
        ib >> make_data(&roi_num_type, sizeof(data_types));
    }

protected:
    std::vector<input_info> get_dependencies() const override {
        std::vector<input_info> ret;
        if (!output_rois_scores.empty())
            ret.push_back(output_rois_scores);
        if (!output_rois_num.empty())
            ret.push_back(output_rois_num);
        return ret;
    }
};
}  // namespace cldnn
