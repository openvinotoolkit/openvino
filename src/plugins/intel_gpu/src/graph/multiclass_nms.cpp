// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>

#include "json_object.h"
#include "multiclass_nms_inst.h"
#include "primitive_type_base.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(multiclass_nms)

layout multiclass_nms_inst::calc_output_layout(const multiclass_nms_node& node, const kernel_impl_params& impl_param) {
    const auto input_layout = impl_param.get_input_layout();
    const auto attrs = impl_param.typed_desc<multiclass_nms>()->attrs;

    const auto num_batches =
        node.has_roisnum() ? node.roisnum().get_output_layout().batch() : node.scores().get_output_layout().batch();
    auto num_classes =
        node.has_roisnum() ? node.boxes().get_output_layout().batch() : node.scores().get_output_layout().feature();
    const auto num_boxes = node.boxes().get_output_layout().feature();

    // see shape_infer() call in MulticlassNmsIEInternal::validate_and_infer_types() - ignore_bg_class == true
    if (attrs.background_class >= 0 && attrs.background_class < num_classes) {
        num_classes = std::max(1, num_classes - 1);
    }

    int max_output_boxes_per_class = 0;
    if (attrs.nms_top_k >= 0) {
        max_output_boxes_per_class = std::min(num_boxes, attrs.nms_top_k);
    } else {
        max_output_boxes_per_class = num_boxes;
    }

    auto max_output_boxes_per_batch = max_output_boxes_per_class * num_classes;
    if (attrs.keep_top_k >= 0) {
        max_output_boxes_per_batch = std::min(max_output_boxes_per_batch, attrs.keep_top_k);
    }

    const auto dim = max_output_boxes_per_batch * num_batches;
    constexpr auto output_size = 6; // 4 coordinates + 1 class + 1 score
    return layout{input_layout.data_type, input_layout.format, {dim, output_size, 1, 1}};
}

std::string multiclass_nms_inst::to_string(const multiclass_nms_node& node) {
    const auto attrs = node.get_primitive()->attrs;
    std::stringstream primitive_description;

    json_composite ed_info;
    ed_info.add("sort_result_type", static_cast<int>(attrs.sort_result));
    ed_info.add("sort_result_across_batch", attrs.sort_result_across_batch);
    ed_info.add("output_type", attrs.indices_output_type);
    ed_info.add("iou_threshold", attrs.iou_threshold);
    ed_info.add("score_threshold", attrs.score_threshold);
    ed_info.add("nms_top_k", attrs.nms_top_k);
    ed_info.add("keep_top_k", attrs.keep_top_k);
    ed_info.add("background_class", attrs.background_class);
    ed_info.add("normalized", attrs.normalized);
    ed_info.add("nms_eta", attrs.nms_eta);

    auto node_info = node.desc_to_json();
    node_info->add("multiclass_nms_info", ed_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}
}  // namespace cldnn
