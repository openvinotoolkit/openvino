// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <experimental_detectron_topk_rois_inst.h>
#include "primitive_type_base.h"
#include <sstream>
#include <json_object.h>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(experimental_detectron_topk_rois)

experimental_detectron_topk_rois_inst::typed_primitive_inst(network& network, experimental_detectron_topk_rois_node const &node)
: parent(network, node) {
}

template<typename ShapeType>
std::vector<layout> experimental_detectron_topk_rois_inst::calc_output_layouts(
        experimental_detectron_topk_rois_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto input_layout = impl_param.get_input_layout();
    auto desc = impl_param.typed_desc<experimental_detectron_topk_rois>();

    return { layout(ov::PartialShape{static_cast<int64_t>(desc->max_rois), 4}, input_layout.data_type, input_layout.format) };
}

template std::vector<layout>
experimental_detectron_topk_rois_inst::calc_output_layouts<ov::PartialShape>(
    experimental_detectron_topk_rois_node const& node, const kernel_impl_params& impl_param);

layout experimental_detectron_topk_rois_inst::calc_output_layout(
    experimental_detectron_topk_rois_node const &node, kernel_impl_params const& impl_param) {
    auto input_layout = impl_param.get_input_layout();
    auto desc = impl_param.typed_desc<experimental_detectron_topk_rois>();

    int32_t roi_num = std::min(input_layout.get_tensor().sizes()[0], static_cast<int32_t>(desc->max_rois));

    return {input_layout.data_type, input_layout.format,  {roi_num,
                                                                 input_layout.get_tensor().sizes()[1], 1, 1 }};
}

std::string experimental_detectron_topk_rois_inst::to_string(experimental_detectron_topk_rois_node const &node) {
    auto node_info = node.desc_to_json();
    json_composite experimental_detectron_topk_rois_info;
    experimental_detectron_topk_rois_info.add("input id", node.input().id());
    experimental_detectron_topk_rois_info.add("indices id", node.input(1).id());
    experimental_detectron_topk_rois_info.add("max_rois", node.get_primitive()->max_rois);
    node_info->add("experimental detectron TopK ROIs info", experimental_detectron_topk_rois_info);
    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

} // namespace cldnn
