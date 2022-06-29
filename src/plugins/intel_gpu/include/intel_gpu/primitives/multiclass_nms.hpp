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

/// @brief multiclass NMS
struct multiclass_nms : public primitive_base<multiclass_nms> {
    CLDNN_DECLARE_PRIMITIVE(multiclass_nms)

    /// @brief Constructs multiclass_nms primitive
    /// @param FIXME opoluektov document
    /// @param
    /// @param
    /// @param
    /// @param
    multiclass_nms(const primitive_id& id,
                   const primitive_id& boxes,
                   const primitive_id& scores,
                   const primitive_id& roisnum,
                   const primitive_id& output_selected_indices,
                   const primitive_id& output_selected_num,
                   int sort_result,  // FIXME opoluektov enum
                   bool sort_result_across_batch,
                   ov::element::Type output_type,
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
          output_type(output_type),
          iou_threshold(iou_threshold),
          score_threshold(score_threshold),
          nms_top_k(nms_top_k),
          keep_top_k(keep_top_k),
          background_class(background_class),
          normalized(normalized),
          nms_eta(nms_eta),
          has_roisnum(!roisnum.empty()) {}

    primitive_id output_selected_indices;
    primitive_id output_selected_num;
    int sort_result;  // FIXME opoluektov enum
    bool sort_result_across_batch;
    ov::element::Type output_type;
    float iou_threshold;
    float score_threshold;
    int nms_top_k;
    int keep_top_k;
    int background_class;
    bool normalized;
    float nms_eta;
    bool has_roisnum;  // FIXME opoluektov: do I need this field?

protected:
    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override {
        std::vector<std::reference_wrapper<const primitive_id>> ret;
        if (!output_selected_indices.empty())
            ret.emplace_back(output_selected_indices);

        if (!output_selected_num.empty())
            ret.emplace_back(output_selected_num);

        return ret;
    }
};

/// @}
/// @}
/// @}
}  // namespace cldnn
