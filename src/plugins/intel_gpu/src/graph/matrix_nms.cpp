// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <json_object.h>

#include <sstream>
#include <string>

#include "matrix_nms_inst.h"
#include "openvino/core/enum_names.hpp"
#include "primitive_type_base.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(matrix_nms)

template<typename ShapeType>
std::vector<layout> matrix_nms_inst::calc_output_layouts(matrix_nms_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto input_layout = impl_param.get_input_layout();
    const auto desc = impl_param.typed_desc<matrix_nms>();
    const auto num_outputs = desc->output_size();

    const auto boxes_ps = input_layout.get_partial_shape();
    const auto scores_ps = impl_param.get_input_layout(1).get_partial_shape();

    auto first_dim_shape = ov::Dimension::dynamic();

    const auto num_boxes_boxes = boxes_ps[1];
    if (num_boxes_boxes.is_static() && scores_ps[0].is_static() && scores_ps[1].is_static()) {
        const auto num_boxes = num_boxes_boxes.get_length();
        auto num_classes = scores_ps[1].get_length();
        if (desc->attribs.background_class >= 0 && desc->attribs.background_class < num_classes) {
            num_classes = std::max(int64_t{1}, num_classes - 1);
        }
        int64_t max_output_boxes_per_class = 0;
        if (desc->attribs.nms_top_k >= 0)
            max_output_boxes_per_class = std::min(num_boxes, static_cast<int64_t>(desc->attribs.nms_top_k));
        else
            max_output_boxes_per_class = num_boxes;

        auto max_output_boxes_per_batch = max_output_boxes_per_class * num_classes;
        if (desc->attribs.keep_top_k >= 0)
            max_output_boxes_per_batch =
                std::min(max_output_boxes_per_batch, static_cast<int64_t>(desc->attribs.keep_top_k));

        first_dim_shape = max_output_boxes_per_batch * scores_ps[0].get_length();
    }

    const auto selected_outputs_shape = ov::PartialShape({first_dim_shape, 6});
    const auto selected_indices_shape = ov::PartialShape({first_dim_shape, 1});
    const auto selected_num_shape = ov::PartialShape({boxes_ps[0]});

    std::vector<ShapeType> out_shapes = {selected_outputs_shape, selected_indices_shape, selected_num_shape};

    std::vector<layout> out_layouts;
    for (size_t i = 0; i < num_outputs; i++) {
        out_layouts.push_back(layout(out_shapes[i], desc->output_data_types[i].value(), format::get_default_format(out_shapes[i].size())));
    }

    return out_layouts;
}

template std::vector<layout> matrix_nms_inst::calc_output_layouts<ov::PartialShape>(matrix_nms_node const& node, const kernel_impl_params& impl_param);

layout matrix_nms_inst::calc_output_layout(const matrix_nms_node& node, const kernel_impl_params& impl_param) {
    const auto primitive = impl_param.typed_desc<matrix_nms>();
    const auto boxes_layout = impl_param.get_input_layout(0);
    const auto scores_layout = impl_param.get_input_layout(1);

    const auto batches_num = boxes_layout.batch();
    auto classes_num = scores_layout.feature();
    const auto boxes_num = boxes_layout.feature();

    if (primitive->attribs.background_class >= 0 && primitive->attribs.background_class < classes_num)
        classes_num = std::max(1, classes_num - 1);

    int max_output_boxes_per_class{boxes_num};
    if (primitive->attribs.nms_top_k >= 0)
        max_output_boxes_per_class = std::min(max_output_boxes_per_class, primitive->attribs.nms_top_k);

    auto max_output_boxes_per_batch = max_output_boxes_per_class * classes_num;
    if (primitive->attribs.keep_top_k >= 0)
        max_output_boxes_per_batch = std::min(max_output_boxes_per_batch, primitive->attribs.keep_top_k);

    auto output_num = max_output_boxes_per_batch * batches_num;

    // BOX_DATA: class_id, box_score, xmin, ymin, xmax, ymax
    constexpr size_t BOX_DATA{6};
    return layout(boxes_layout.data_type, boxes_layout.format, {output_num, BOX_DATA, 1, 1});
}

std::string matrix_nms_inst::to_string(const matrix_nms_node& node) {
    json_composite matrix_nms_info;
    matrix_nms_info.add("boxes id", node.input().id());
    matrix_nms_info.add("scores id", node.get_dependency(1).id());
    matrix_nms_info.add("sort_result_type", ov::as_string(node.get_primitive()->attribs.sort_result_type));
    matrix_nms_info.add("decay_function", ov::as_string(node.get_primitive()->attribs.decay_function));
    matrix_nms_info.add("sort_result_across_batch", node.get_primitive()->attribs.sort_result_across_batch);
    matrix_nms_info.add("score_threshold", node.get_primitive()->attribs.score_threshold);
    matrix_nms_info.add("nms_top_k", node.get_primitive()->attribs.nms_top_k);
    matrix_nms_info.add("keep_top_k", node.get_primitive()->attribs.keep_top_k);
    matrix_nms_info.add("background_class", node.get_primitive()->attribs.background_class);
    matrix_nms_info.add("gaussian_sigma", node.get_primitive()->attribs.gaussian_sigma);
    matrix_nms_info.add("post_threshold", node.get_primitive()->attribs.post_threshold);
    matrix_nms_info.add("normalized", node.get_primitive()->attribs.normalized);

    auto node_info = node.desc_to_json();
    node_info->add("matrix_nms info", matrix_nms_info);
    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

}  // namespace cldnn
