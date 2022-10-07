// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <utility>
#include <vector>

#include "openvino/core/type/element_type.hpp"
#include "primitive.hpp"

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

enum class sort_result_type : int32_t {
    classid,  // sort selected boxes by class id (ascending) in each batch element
    score,    // sort selected boxes by score (descending) in each batch element
    none      // do not guarantee the order in each batch element
};


/// @brief multiclass NMS
struct multiclass_nms : public primitive_base<multiclass_nms> {
    CLDNN_DECLARE_PRIMITIVE(multiclass_nms)

    /// @brief Constructs multiclass_nms primitive
    /// @param id This primitive id
    /// @param boxes Boxes coordinates
    /// @param scores Box scores
    /// @param roisnum Number of boxes in each batch for MulticlassNMS-9 (empty string for MulticlassNMS-8)
    /// @param output_selected_indices Indices of selected boxes
    /// @param output_selected_num Number of selected boxes in each batch
    /// @param sort_result Sort mode (by class, by score, do not sort)
    /// @param sort_result_across_batch If true, selected boxes will be sorted across all batches
    /// @param output_type integer type for output indices and numbers of boxes
    /// @param iou_threshold Threshold for intersection over union
    /// @param score_threshold Minimum score to process a box
    /// @param nms_top_k Max number of boxes to be selected per class
    /// @param keep_top_k Max number of boxes to be selected per batch
    /// @param background_class Background class id
    /// @param normalized If true, box coordinates are considered as normalized
    /// @param nms_eta Parameter for adaptive non-max-suppression
    multiclass_nms(const primitive_id& id,
                   const primitive_id& boxes,
                   const primitive_id& scores,
                   const primitive_id& roisnum,
                   const primitive_id& output_selected_indices,
                   const primitive_id& output_selected_num,
                   sort_result_type sort_result,
                   bool sort_result_across_batch,
                   data_types output_type,
                   float iou_threshold,
                   float score_threshold,
                   int nms_top_k,
                   int keep_top_k,
                   int background_class,
                   bool normalized,
                   float nms_eta,
                   const primitive_id& ext_prim_id = "",
                   const padding& output_padding = {})
        : primitive_base{id,
                         roisnum.empty()
                             ? std::vector<primitive_id>({boxes, scores, output_selected_indices, output_selected_num})
                             : std::vector<primitive_id>(
                                   {boxes, scores, roisnum, output_selected_indices, output_selected_num}),
                         output_padding},
          output_selected_indices(output_selected_indices),
          output_selected_num(output_selected_num),
          sort_result(sort_result),
          sort_result_across_batch(sort_result_across_batch),
          indices_output_type(output_type),
          iou_threshold(iou_threshold),
          score_threshold(score_threshold),
          nms_top_k(nms_top_k),
          keep_top_k(keep_top_k),
          background_class(background_class),
          normalized(normalized),
          nms_eta(nms_eta),
          has_roisnum(!roisnum.empty()) {}

    primitive_id output_selected_indices{};
    primitive_id output_selected_num{};
    sort_result_type sort_result{sort_result_type::none};
    bool sort_result_across_batch{false};
    data_types indices_output_type{data_types::i64};
    float iou_threshold{0.0f};
    float score_threshold{0.0f};
    int nms_top_k{-1};
    int keep_top_k{-1};
    int background_class{-1};
    bool normalized{true};
    float nms_eta{1.0f};
    bool has_roisnum{false};

protected:
    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override {
        std::vector<std::reference_wrapper<const primitive_id>> ret;
        ret.emplace_back(output_selected_indices);
        ret.emplace_back(output_selected_num);
        return ret;
    }
};

/// @}
/// @}
/// @}
}  // namespace cldnn
