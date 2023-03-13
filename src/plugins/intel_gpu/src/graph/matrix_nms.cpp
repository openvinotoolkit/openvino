// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <json_object.h>

#include <sstream>
#include <string>

#include "matrix_nms_inst.h"
#include "openvino/core/enum_names.hpp"
#include "primitive_type_base.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(matrix_nms)

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
    matrix_nms_info.add("sort_result_type", ov::as_string(node.get_primitive()->attribs.sort_type));
    matrix_nms_info.add("decay_function", ov::as_string(node.get_primitive()->attribs.decay));
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

namespace ov {
using cldnn::matrix_nms;

template <>
EnumNames<matrix_nms::decay_function>& EnumNames<matrix_nms::decay_function>::get() {
    static auto enum_names = EnumNames<matrix_nms::decay_function>(
        "decay_function",
        {{"gaussian", matrix_nms::decay_function::gaussian}, {"linear", matrix_nms::decay_function::linear}});
    return enum_names;
}

template <>
EnumNames<matrix_nms::sort_result_type>& EnumNames<matrix_nms::sort_result_type>::get() {
    static auto enum_names =
        EnumNames<matrix_nms::sort_result_type>("sort_result_type",
                                                {{"class_id", matrix_nms::sort_result_type::class_id},
                                                 {"score", matrix_nms::sort_result_type::score},
                                                 {"none", matrix_nms::sort_result_type::none}});
    return enum_names;
}
}  // namespace ov
