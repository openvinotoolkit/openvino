// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <utility>
#include <vector>

#include "ngraph/op/multiclass_nms.hpp"
#include "openvino/core/type/element_type.hpp"
#include "primitive.hpp"

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief multiclass NMS
struct multiclass_nms : public primitive_base<multiclass_nms> {
    CLDNN_DECLARE_PRIMITIVE(multiclass_nms)

    enum class sort_result_type : int32_t {
        classid,  // sort selected boxes by class id (ascending) in each batch element
        score,    // sort selected boxes by score (descending) in each batch element
        none      // do not guarantee the order in each batch element
    };

    struct attributes {
        // Sort mode (by class, by score, do not sort)
        sort_result_type sort_result{sort_result_type::none};
        // If true, selected boxes will be sorted across all batches
        bool sort_result_across_batch{false};
        // Integer type for output indices and numbers of boxes
        data_types indices_output_type{data_types::i64};
        // Threshold for intersection over union
        float iou_threshold{0.0f};
        // Minimum score to process a box
        float score_threshold{0.0f};
        // Max number of boxes to be selected per class
        int nms_top_k{-1};
        // Max number of boxes to be selected per batch
        int keep_top_k{-1};
        // Background class id
        int background_class{-1};
        // If true, box coordinates are considered as normalized
        bool normalized{true};
        // Parameter for adaptive non-max-suppression
        float nms_eta{1.0f};

        attributes() = default;

        attributes(sort_result_type sort_result,
                   bool sort_result_across_batch,
                   data_types indices_output_type,
                   float iou_threshold,
                   float score_threshold,
                   int nms_top_k,
                   int keep_top_k,
                   int background_class,
                   bool normalized,
                   float nms_eta)
            : sort_result(sort_result),
              sort_result_across_batch(sort_result_across_batch),
              indices_output_type(indices_output_type),
              iou_threshold(iou_threshold),
              score_threshold(score_threshold),
              nms_top_k(nms_top_k),
              keep_top_k(keep_top_k),
              background_class(background_class),
              normalized(normalized),
              nms_eta(nms_eta) {}

        attributes(const ngraph::op::util::MulticlassNmsBase::Attributes& attrs)
            : attributes(from(attrs.sort_result_type),
                         attrs.sort_result_across_batch,
                         cldnn::element_type_to_data_type(attrs.output_type),
                         attrs.iou_threshold,
                         attrs.score_threshold,
                         attrs.nms_top_k,
                         attrs.keep_top_k,
                         attrs.background_class,
                         attrs.normalized,
                         attrs.nms_eta) {}

    private:
        static sort_result_type from(const ngraph::op::util::MulticlassNmsBase::SortResultType sort_result_type) {
            switch (sort_result_type) {
                case ngraph::op::util::MulticlassNmsBase::SortResultType::CLASSID:
                    return sort_result_type::classid;
                case ngraph::op::util::MulticlassNmsBase::SortResultType::SCORE:
                    return sort_result_type::score;
                case ngraph::op::util::MulticlassNmsBase::SortResultType::NONE:
                    return sort_result_type::none;
                default:
                    return sort_result_type::none;
            }
        }
    };
    /// @brief Constructs multiclass_nms primitive
    /// @param id This primitive id
    /// @param boxes Boxes coordinates
    /// @param scores Box scores
    /// @param roisnum Number of boxes in each batch for MulticlassNMS-9 (empty string for MulticlassNMS-8)
    /// @param output_selected_indices Indices of selected boxes
    /// @param output_selected_num Number of selected boxes in each batch
    /// @param attrs Attributes
    /// @param nms_eta Parameter for adaptive non-max-suppression
    multiclass_nms(const primitive_id& id,
                   const std::vector<input_info> inputs,
                   const multiclass_nms::attributes& attrs,
                   const primitive_id& ext_prim_id = "",
                   const padding& output_padding = {})
        : primitive_base{id,
                         inputs[InputIdx::RoisNum].pid.empty()
                             ? std::vector<input_info>({inputs[InputIdx::Boxes],
                                                        inputs[InputIdx::Scores],
                                                        inputs[InputIdx::OutputSelectedIndices],
                                                        inputs[InputIdx::OutputSelectedNum]})
                             : inputs,
                         {output_padding}},
          output_selected_indices(inputs[InputIdx::OutputSelectedIndices].pid),
          output_selected_num(inputs[InputIdx::OutputSelectedNum].pid),
          attrs(attrs),
          has_roisnum(!inputs[InputIdx::RoisNum].pid.empty()) {}

    primitive_id output_selected_indices{};
    primitive_id output_selected_num{};
    attributes attrs;
    bool has_roisnum{false};

protected:
    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override {
        std::vector<std::reference_wrapper<const primitive_id>> ret;
        ret.emplace_back(output_selected_indices);
        ret.emplace_back(output_selected_num);
        return ret;
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

/// @}
/// @}
/// @}
}  // namespace cldnn
