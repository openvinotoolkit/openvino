// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <search_sorted_inst.h>
#include "primitive_type_base.h"
#include <sstream>
#include <json_object.h>
#include "openvino/core/enum_names.hpp"
#include "search_sorted_shape_inference.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(search_sorted)

search_sorted_inst::typed_primitive_inst(network& network, search_sorted_node const& node)
    : parent(network, node) {}

layout search_sorted_inst::calc_output_layout(search_sorted_node const& node, kernel_impl_params const& impl_param) {
    auto primitive = impl_param.typed_desc<search_sorted>();
    auto input_layout = impl_param.get_input_layout(0);
    return layout();
}

template<typename ShapeType>
std::vector<layout> search_sorted_inst::calc_output_layouts(search_sorted_node const& node, kernel_impl_params const& impl_param) {
    return std::vector<layout>();
}

std::string search_sorted_inst::to_string(search_sorted_node const& node) {
    auto node_info = node.desc_to_json();
    json_composite search_sorted_info;
    // search_sorted_info.add("input id", node.input().id());
    // search_sorted_info.add("rois id", node.get_dependency(1).id());
    // search_sorted_info.add("batches id", node.get_dependency(2).id());
    // search_sorted_info.add("pooled_h", node.get_primitive()->pooled_h);
    // search_sorted_info.add("pooled_w", node.get_primitive()->pooled_w);
    // search_sorted_info.add("sampling_ratio", node.get_primitive()->sampling_ratio);
    // search_sorted_info.add("spatial_scale", node.get_primitive()->spatial_scale);
    // search_sorted_info.add("pooling_mode", ov::as_string(node.get_primitive()->pooling_mode));
    // search_sorted_info.add("aligned_mode", ov::as_string(node.get_primitive()->aligned_mode));
    node_info->add("search_sorted info", search_sorted_info);
    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

}  // namespace cldnn