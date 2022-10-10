// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <vector>

#include "ngraph/op/matrix_nms.hpp"
#include "primitive.hpp"

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Performs matrix nms of input boxes and returns indices of selected boxes.
struct matrix_nms : public primitive_base<matrix_nms> {
    CLDNN_DECLARE_PRIMITIVE(matrix_nms)

    enum DecayFunction { gaussian, linear };

    enum SortResultType {
        class_id,  // sort selected boxes by class id (ascending) in each batch element
        score,     // sort selected boxes by score (descending) in each batch element
        none       // do not guarantee the order in each batch element
    };

    /// \brief Structure that specifies attributes of the operation
    struct attributes {
        // specifies order of output elements
        SortResultType sort_result_type = SortResultType::none;
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
        DecayFunction decay_function = DecayFunction::linear;
        // specifies gaussian_sigma parameter for gaussian decay_function
        float gaussian_sigma = 2.0f;
        // specifies threshold to filter out boxes with low confidence score after
        // decaying
        float post_threshold = 0.0f;
        // specifies whether boxes are normalized or not
        bool normalized = true;

        attributes() {}

        attributes(const ngraph::op::v8::MatrixNms::Attributes& attrs)
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

        attributes(SortResultType sort_result_type,
                   bool sort_result_across_batch,
                   float score_threshold,
                   int nms_top_k,
                   int keep_top_k,
                   int background_class,
                   DecayFunction decay_function,
                   float gaussian_sigma,
                   float post_threshold,
                   bool normalized)
            : sort_result_type(sort_result_type),
              sort_result_across_batch(sort_result_across_batch),
              score_threshold(score_threshold),
              nms_top_k(nms_top_k),
              keep_top_k(keep_top_k),
              background_class(background_class),
              decay_function(decay_function),
              gaussian_sigma(gaussian_sigma),
              post_threshold(post_threshold),
              normalized(normalized) {}
    };

    /// @brief Constructs matrix_nms primitive.
    /// @param id This primitive id.
    /// @param boxes primitive id.
    /// @param scores primitive id.
    /// @param second_output primitive id.
    /// @param third_output primitive id.
    /// @param attrs attributes.
    matrix_nms(const primitive_id& id,
               const primitive_id& boxes,
               const primitive_id& scores,
               const primitive_id& second_output,
               const primitive_id& third_output,
               const matrix_nms::attributes& attrs)
        : primitive_base(id, {boxes, scores}),
          second_output(second_output),
          third_output(third_output),
          attribs(attrs) {}

    /// @brief Constructs matrix_nms primitive.
    /// @param id This primitive id.
    /// @param boxes primitive id.
    /// @param scores primitive id.
    /// @param second_output primitive id.
    /// @param third_output primitive id.
    /// @param attrs operation attributes.
    matrix_nms(const primitive_id& id,
               const primitive_id& boxes,
               const primitive_id& scores,
               const primitive_id& second_output,
               const primitive_id& third_output,
               const ngraph::op::v8::MatrixNms::Attributes& attrs)
        : primitive_base(id, {boxes, scores}),
          second_output(second_output),
          third_output(third_output),
          attribs(attrs) {}

    primitive_id second_output;
    primitive_id third_output;

    attributes attribs;

    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override {
        std::vector<std::reference_wrapper<const primitive_id>> ret;
        ret.push_back(second_output);
        ret.push_back(third_output);

        return ret;
    }

private:
    static cldnn::matrix_nms::DecayFunction from(ngraph::op::v8::MatrixNms::DecayFunction decay) {
        switch (decay) {
        case ngraph::op::v8::MatrixNms::DecayFunction::GAUSSIAN:
            return cldnn::matrix_nms::DecayFunction::gaussian;
        case ngraph::op::v8::MatrixNms::DecayFunction::LINEAR:
        default:
            return cldnn::matrix_nms::DecayFunction::linear;
        }
    }

    static cldnn::matrix_nms::SortResultType from(ngraph::op::v8::MatrixNms::SortResultType type) {
        switch (type) {
        case ngraph::op::v8::MatrixNms::SortResultType::CLASSID:
            return cldnn::matrix_nms::SortResultType::class_id;
        case ngraph::op::v8::MatrixNms::SortResultType::SCORE:
            return cldnn::matrix_nms::SortResultType::score;
        case ngraph::op::v8::MatrixNms::SortResultType::NONE:
        default:
            return cldnn::matrix_nms::SortResultType::none;
        }
    }
};
/// @}
/// @}
/// @}
}  // namespace cldnn
