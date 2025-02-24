// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/multiclass_nms.hpp"
#include "openvino/core/type/element_type.hpp"
#include "primitive.hpp"

#include <utility>
#include <vector>

namespace cldnn {

/// @brief multiclass NMS
struct multiclass_nms : public primitive_base<multiclass_nms> {
    CLDNN_DECLARE_PRIMITIVE(multiclass_nms)

    multiclass_nms() : primitive_base("", {}) {}

    /// @brief Constructs multiclass_nms primitive
    /// @param id This primitive id
    /// @param boxes Boxes coordinates
    /// @param scores Box scores
    /// @param roisnum Number of boxes in each batch for MulticlassNMS-9 (empty string for MulticlassNMS-8)
    /// @param attrs Attributes
    multiclass_nms(const primitive_id& id,
                   const std::vector<input_info> inputs,
                   const ov::op::util::MulticlassNmsBase::Attributes& attrs,
                   const padding& output_padding = {})
        : primitive_base{id, inputs},
          attrs(attrs) {}

    ov::op::util::MulticlassNmsBase::Attributes attrs;

   size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, attrs.background_class);
        seed = hash_combine(seed, attrs.iou_threshold);
        seed = hash_combine(seed, attrs.keep_top_k);
        seed = hash_combine(seed, attrs.nms_eta);
        seed = hash_combine(seed, attrs.nms_top_k);
        seed = hash_combine(seed, attrs.normalized);
        seed = hash_combine(seed, attrs.score_threshold);
        seed = hash_combine(seed, attrs.sort_result_type);
        seed = hash_combine(seed, attrs.sort_result_across_batch);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const multiclass_nms>(rhs);

        #define cmp_fields(name) name == rhs_casted.name
        return cmp_fields(attrs.background_class) &&
               cmp_fields(attrs.iou_threshold) &&
               cmp_fields(attrs.keep_top_k) &&
               cmp_fields(attrs.nms_eta) &&
               cmp_fields(attrs.nms_top_k) &&
               cmp_fields(attrs.normalized) &&
               cmp_fields(attrs.score_threshold) &&
               cmp_fields(attrs.sort_result_type) &&
               cmp_fields(attrs.sort_result_across_batch);
        #undef cmp_fields
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<multiclass_nms>::save(ob);
        ob << make_data(&attrs.sort_result_type, sizeof(ov::op::util::MulticlassNmsBase::SortResultType));
        ob << attrs.sort_result_across_batch;
        ob << attrs.iou_threshold;
        ob << attrs.score_threshold;
        ob << attrs.nms_top_k;
        ob << attrs.keep_top_k;
        ob << attrs.background_class;
        ob << attrs.normalized;
        ob << attrs.nms_eta;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<multiclass_nms>::load(ib);
        ib >> make_data(&attrs.sort_result_type, sizeof(ov::op::util::MulticlassNmsBase::SortResultType));
        ib >> attrs.sort_result_across_batch;
        ib >> attrs.iou_threshold;
        ib >> attrs.score_threshold;
        ib >> attrs.nms_top_k;
        ib >> attrs.keep_top_k;
        ib >> attrs.background_class;
        ib >> attrs.normalized;
        ib >> attrs.nms_eta;
    }

private:
    enum InputIdx : size_t {
        Boxes = 0,
        Scores = 1,
        RoisNum = 2,
        OutputSelectedIndices = 3,
        OutputSelectedNum = 4,
    };
};

}  // namespace cldnn
