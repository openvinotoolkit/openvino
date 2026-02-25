// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>

#include "experimental_detectron_detection_output_inst.hpp"
#include "json_object.h"
#include "primitive_type_base.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(experimental_detectron_detection_output)

template<typename ShapeType>
std::vector<layout> experimental_detectron_detection_output_inst::calc_output_layouts(experimental_detectron_detection_output_node const& /*node*/,
                                                                                      const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<experimental_detectron_detection_output>();

    const ov::PartialShape boxes_shape = {static_cast<int64_t>(desc->max_detections_per_image), 4};
    const ov::PartialShape classes_shape = {static_cast<int64_t>(desc->max_detections_per_image)};
    const ov::PartialShape scores_shape = {static_cast<int64_t>(desc->max_detections_per_image)};

    std::vector<ShapeType> out_shapes = { boxes_shape, classes_shape, scores_shape };
    std::vector<layout> out_layouts;
    for (size_t i = 0; i < desc->output_size(); i++) {
        out_layouts.push_back(layout(out_shapes[i].get_max_shape(), desc->output_data_types[i].value(), format::get_default_format(out_shapes[i].size())));
    }

    return out_layouts;
}

template std::vector<layout>
experimental_detectron_detection_output_inst::calc_output_layouts<ov::PartialShape>(
        experimental_detectron_detection_output_node const& node, const kernel_impl_params& impl_param);

layout experimental_detectron_detection_output_inst::calc_output_layout(
    const experimental_detectron_detection_output_node& node, kernel_impl_params const& impl_param) {
    const layout data_layout = impl_param.get_input_layout();
    auto desc = impl_param.typed_desc<experimental_detectron_detection_output>();

    return layout(data_layout.data_type, data_layout.format, {static_cast<int>(desc->max_detections_per_image), 4, 1, 1});
}

std::string experimental_detectron_detection_output_inst::to_string(
    const experimental_detectron_detection_output_node& node) {
    auto desc = node.get_primitive();

    std::stringstream primitive_description;

    json_composite ed_info;
    ed_info.add("score_threshold", desc->score_threshold);
    ed_info.add("nms_threshold", desc->nms_threshold);
    ed_info.add("score_threshold", desc->score_threshold);
    ed_info.add("max_delta_log_wh", desc->max_delta_log_wh);
    ed_info.add("num_classes", desc->num_classes);
    ed_info.add("post_nms_count", desc->post_nms_count);
    ed_info.add("max_detections_per_image", desc->max_detections_per_image);
    ed_info.add("class_agnostic_box_regression", desc->class_agnostic_box_regression);
    ed_info.add("deltas_weights", desc->deltas_weights);

    auto node_info = node.desc_to_json();
    node_info->add("experimental_detectron_detection_output_info", ed_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}
}  // namespace cldnn
