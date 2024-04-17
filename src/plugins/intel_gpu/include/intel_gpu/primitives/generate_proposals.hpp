// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "openvino/op/generate_proposals.hpp"
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
    /// @param inputs input primitive ids
    /// @param attr Attributes of GenerateProposal op
    generate_proposals(const primitive_id& id,
                       const std::vector<input_info>& inputs,
                       const ov::op::v9::GenerateProposals::Attributes& attrs) :
            primitive_base{id, inputs},
            attrs{attrs} {}

    ov::op::v9::GenerateProposals::Attributes attrs;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, attrs.min_size);
        seed = hash_combine(seed, attrs.nms_threshold);
        seed = hash_combine(seed, attrs.pre_nms_count);
        seed = hash_combine(seed, attrs.post_nms_count);
        seed = hash_combine(seed, attrs.normalized);
        seed = hash_combine(seed, attrs.nms_eta);
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
               cmp_fields(attrs.nms_eta);
        #undef cmp_fields
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<generate_proposals>::save(ob);
        ob << attrs.min_size;
        ob << attrs.nms_threshold;
        ob << attrs.pre_nms_count;
        ob << attrs.post_nms_count;
        ob << attrs.normalized;
        ob << attrs.nms_eta;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<generate_proposals>::load(ib);
        ib >> attrs.min_size;
        ib >> attrs.nms_threshold;
        ib >> attrs.pre_nms_count;
        ib >> attrs.post_nms_count;
        ib >> attrs.normalized;
        ib >> attrs.nms_eta;
    }
};
}  // namespace cldnn
