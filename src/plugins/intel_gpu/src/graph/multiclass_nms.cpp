// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>

#include "multiclass_nms_inst.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include "primitive_type_base.h"

namespace cldnn {
primitive_type_id multiclass_nms::type_id() {
    static primitive_type_base<multiclass_nms> instance;
    return &instance;
}

layout multiclass_nms_inst::calc_output_layout(
    const multiclass_nms_node& node, const kernel_impl_params& impl_param) {
    const auto input_layout = impl_param.get_input_layout();
    const auto desc = impl_param.typed_desc<multiclass_nms>();

    const auto num_batches = node.has_roisnum() ? node.roisnum().get_output_layout().batch() : node.scores().get_output_layout().batch();
    auto num_classes = node.has_roisnum() ? node.boxes().get_output_layout().batch() : node.scores().get_output_layout().feature();
    const auto num_boxes = node.boxes().get_output_layout().feature();

    // see shape_infer() call in MulticlassNmsIEInternal::validate_and_infer_types() - ignore_bg_class == true
   if (desc->background_class >= 0 && desc->background_class < num_classes) {
        num_classes = std::max(1, num_classes - 1);
    }

    int max_output_boxes_per_class = 0;
    if (desc->nms_top_k >= 0)
        max_output_boxes_per_class = std::min(num_boxes, desc->nms_top_k);
    else
        max_output_boxes_per_class = num_boxes;

    auto max_output_boxes_per_batch = max_output_boxes_per_class * num_classes;
    if (desc->keep_top_k >= 0)
        max_output_boxes_per_batch = std::min(max_output_boxes_per_batch, desc->keep_top_k);

    const auto dim = max_output_boxes_per_batch * num_batches;

    return layout{input_layout.data_type, input_layout.format, {dim, 6, 1, 1}};
}

std::string multiclass_nms_inst::to_string(
    const multiclass_nms_node& node) {
    const auto desc = node.get_primitive();

    std::stringstream primitive_description;

    json_composite ed_info;
    ed_info.add("sort_result_type", static_cast<int>(desc->sort_result));
    ed_info.add("sort_result_across_batch", desc->sort_result_across_batch);
    ed_info.add("output_type", desc->indices_output_type);
    ed_info.add("iou_threshold", desc->iou_threshold);
    ed_info.add("score_threshold", desc->score_threshold);
    ed_info.add("nms_top_k", desc->nms_top_k);
    ed_info.add("keep_top_k", desc->keep_top_k);
    ed_info.add("background_class", desc->background_class);
    ed_info.add("normalized", desc->normalized);
    ed_info.add("nms_eta", desc->nms_eta);

    auto node_info = node.desc_to_json();
    node_info->add("multiclass_nms_info", ed_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}
}  // namespace cldnn
