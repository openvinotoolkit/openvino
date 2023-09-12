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

    enum decay_function { gaussian, linear };

    enum sort_result_type {
        class_id,  // sort selected boxes by class id (ascending) in each batch element
        score,     // sort selected boxes by score (descending) in each batch element
        none       // do not guarantee the order in each batch element
    };

    /// \brief Structure that specifies attributes of the operation
    struct attributes {
        // specifies order of output elements
        sort_result_type sort_type = sort_result_type::none;
        // specifies whenever it is necessary to sort selected boxes across batches or not
        bool sort_result_across_batch = false;
        // specifies minimum score to consider box for the processing
        float score_threshold = 0.0f;
        // specifies maximum number of boxes to be selected per class, -1 meaning to
        // keep all boxes
        int nms_top_k = -1;
        // specifies maximum number of boxes to be selected per batch element, -1
        // meaning to keep all boxes
        int keep_top_k = -1;
        // specifies the background class id, -1 meaning to keep all classes
        int background_class = -1;
        // specifies decay function used to decay scores
        decay_function decay = decay_function::linear;
        // specifies gaussian_sigma parameter for gaussian decay_function
        float gaussian_sigma = 2.0f;
        // specifies threshold to filter out boxes with low confidence score after
        // decaying
        float post_threshold = 0.0f;
        // specifies whether boxes are normalized or not
        bool normalized = true;

        attributes() {}

        attributes(const ov::op::v8::MatrixNms::Attributes& attrs)
            : attributes(from(attrs.sort_result_type),
                         attrs.sort_result_across_batch,
                         attrs.score_threshold,
                         attrs.nms_top_k,
                         attrs.keep_top_k,
                         attrs.background_class,
                         from(attrs.decay_function),
                         attrs.gaussian_sigma,
                         attrs.post_threshold,
                         attrs.normalized) {}

        attributes(sort_result_type sort_type,
                   bool sort_result_across_batch,
                   float score_threshold,
                   int nms_top_k,
                   int keep_top_k,
                   int background_class,
                   decay_function decay,
                   float gaussian_sigma,
                   float post_threshold,
                   bool normalized)
            : sort_type(sort_type),
              sort_result_across_batch(sort_result_across_batch),
              score_threshold(score_threshold),
              nms_top_k(nms_top_k),
              keep_top_k(keep_top_k),
              background_class(background_class),
              decay(decay),
              gaussian_sigma(gaussian_sigma),
              post_threshold(post_threshold),
              normalized(normalized) {}

        void save(BinaryOutputBuffer& ob) const {
            ob << make_data(&sort_type, sizeof(sort_result_type));
            ob << sort_result_across_batch;
            ob << score_threshold;
            ob << nms_top_k;
            ob << keep_top_k;
            ob << background_class;
            ob << make_data(&decay, sizeof(decay_function));
            ob << gaussian_sigma;
            ob << post_threshold;
            ob << normalized;
        }

        void load(BinaryInputBuffer& ib) {
            ib >> make_data(&sort_type, sizeof(sort_result_type));
            ib >> sort_result_across_batch;
            ib >> score_threshold;
            ib >> nms_top_k;
            ib >> keep_top_k;
            ib >> background_class;
            ib >> make_data(&decay, sizeof(decay_function));
            ib >> gaussian_sigma;
            ib >> post_threshold;
            ib >> normalized;
        }
    };

    /// @brief Constructs matrix_nms primitive.
    /// @param id This primitive id.
    /// @param boxes primitive id.
    /// @param scores primitive id.
    /// @param second_output primitive id.
    /// @param third_output primitive id.
    /// @param attrs attributes.
    matrix_nms(const primitive_id& id,
               const input_info& boxes,
               const input_info& scores,
               const input_info& second_output,
               const input_info& third_output,
               const matrix_nms::attributes& attrs)
        : primitive_base(id, {boxes, scores, second_output, third_output}),
          attribs(attrs) {}

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

    attributes attribs;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, attribs.sort_type);
        seed = hash_combine(seed, attribs.sort_result_across_batch);
        seed = hash_combine(seed, attribs.score_threshold);
        seed = hash_combine(seed, attribs.nms_top_k);
        seed = hash_combine(seed, attribs.keep_top_k);
        seed = hash_combine(seed, attribs.background_class);
        seed = hash_combine(seed, attribs.decay);
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
        return cmp_fields(attribs.sort_type) &&
               cmp_fields(attribs.sort_result_across_batch) &&
               cmp_fields(attribs.score_threshold) &&
               cmp_fields(attribs.nms_top_k) &&
               cmp_fields(attribs.keep_top_k) &&
               cmp_fields(attribs.background_class) &&
               cmp_fields(attribs.decay) &&
               cmp_fields(attribs.gaussian_sigma) &&
               cmp_fields(attribs.post_threshold) &&
               cmp_fields(attribs.normalized);
        #undef cmp_fields
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<matrix_nms>::save(ob);
        ob << attribs;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<matrix_nms>::load(ib);
        ib >> attribs;
    }

private:
    static cldnn::matrix_nms::decay_function from(ov::op::v8::MatrixNms::DecayFunction decay) {
        switch (decay) {
        case ov::op::v8::MatrixNms::DecayFunction::GAUSSIAN:
            return cldnn::matrix_nms::decay_function::gaussian;
        case ov::op::v8::MatrixNms::DecayFunction::LINEAR:
        default:
            return cldnn::matrix_nms::decay_function::linear;
        }
    }

    static cldnn::matrix_nms::sort_result_type from(ov::op::v8::MatrixNms::SortResultType type) {
        switch (type) {
        case ov::op::v8::MatrixNms::SortResultType::CLASSID:
            return cldnn::matrix_nms::sort_result_type::class_id;
        case ov::op::v8::MatrixNms::SortResultType::SCORE:
            return cldnn::matrix_nms::sort_result_type::score;
        case ov::op::v8::MatrixNms::SortResultType::NONE:
        default:
            return cldnn::matrix_nms::sort_result_type::none;
        }
    }
};
}  // namespace cldnn
