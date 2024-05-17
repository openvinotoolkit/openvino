// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/matrix_nms.hpp"
#include "primitive.hpp"
#include <vector>

namespace cldnn {

/// @brief Performs matrix nms of input boxes and returns indices of selected boxes.
struct matrix_nms : public primitive_base<matrix_nms> {
    CLDNN_DECLARE_PRIMITIVE(matrix_nms)

    matrix_nms() : primitive_base("", {}) {}

    /// @brief Constructs matrix_nms primitive.
    /// @param id This primitive id.
    /// @param boxes primitive id.
    /// @param scores primitive id.
    /// @param second_output primitive id.
    /// @param third_output primitive id.
    /// @param attrs operation attributes.
    matrix_nms(const primitive_id& id,
               const input_info& boxes,
               const input_info& scores,
               const input_info& second_output,
               const input_info& third_output,
               const ov::op::v8::MatrixNms::Attributes& attrs)
        : primitive_base(id, {boxes, scores, second_output, third_output}),
          attribs(attrs) {}

    matrix_nms(const primitive_id& id,
               const input_info& boxes,
               const input_info& scores,
               const ov::op::v8::MatrixNms::Attributes& attrs)
        : primitive_base(id, {boxes, scores}),
          attribs(attrs) {}

    ov::op::v8::MatrixNms::Attributes attribs;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, attribs.sort_result_type);
        seed = hash_combine(seed, attribs.sort_result_across_batch);
        seed = hash_combine(seed, attribs.score_threshold);
        seed = hash_combine(seed, attribs.nms_top_k);
        seed = hash_combine(seed, attribs.keep_top_k);
        seed = hash_combine(seed, attribs.background_class);
        seed = hash_combine(seed, attribs.decay_function);
        seed = hash_combine(seed, attribs.gaussian_sigma);
        seed = hash_combine(seed, attribs.post_threshold);
        seed = hash_combine(seed, attribs.normalized);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const matrix_nms>(rhs);

        #define cmp_fields(name) name == rhs_casted.name
        return cmp_fields(attribs.sort_result_type) &&
               cmp_fields(attribs.sort_result_across_batch) &&
               cmp_fields(attribs.score_threshold) &&
               cmp_fields(attribs.nms_top_k) &&
               cmp_fields(attribs.keep_top_k) &&
               cmp_fields(attribs.background_class) &&
               cmp_fields(attribs.decay_function) &&
               cmp_fields(attribs.gaussian_sigma) &&
               cmp_fields(attribs.post_threshold) &&
               cmp_fields(attribs.normalized);
        #undef cmp_fields
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<matrix_nms>::save(ob);
        ob << make_data(&attribs.sort_result_type, sizeof(ov::op::v8::MatrixNms::SortResultType));
        ob << attribs.sort_result_across_batch;
        ob << attribs.score_threshold;
        ob << attribs.nms_top_k;
        ob << attribs.keep_top_k;
        ob << attribs.background_class;
        ob << make_data(&attribs.decay_function, sizeof(ov::op::v8::MatrixNms::DecayFunction));
        ob << attribs.gaussian_sigma;
        ob << attribs.post_threshold;
        ob << attribs.normalized;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<matrix_nms>::load(ib);
        ib >> make_data(&attribs.sort_result_type, sizeof(ov::op::v8::MatrixNms::SortResultType));
        ib >> attribs.sort_result_across_batch;
        ib >> attribs.score_threshold;
        ib >> attribs.nms_top_k;
        ib >> attribs.keep_top_k;
        ib >> attribs.background_class;
        ib >> make_data(&attribs.decay_function, sizeof(ov::op::v8::MatrixNms::DecayFunction));
        ib >> attribs.gaussian_sigma;
        ib >> attribs.post_threshold;
        ib >> attribs.normalized;
    }
};
}  // namespace cldnn
