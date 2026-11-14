// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <histc_inst.h>
#include <json_object.h>

#include <sstream>

#include "primitive_type_base.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(histc)

histc_inst::typed_primitive_inst(network& network, const histc_node& node) : parent(network, node) {}

layout histc_inst::calc_output_layout(const histc_node& node, const kernel_impl_params& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template <typename ShapeType>
std::vector<layout> histc_inst::calc_output_layouts(const histc_node& /*node*/, const kernel_impl_params& impl_param) {
    const auto desc = impl_param.typed_desc<histc>();
    const auto output_type = desc->output_data_types[0].value_or(impl_param.get_input_layout(0).data_type);
    return {layout{ShapeType{desc->bins}, output_type, format::bfyx}};
}

template std::vector<layout> histc_inst::calc_output_layouts<ov::PartialShape>(const histc_node& node, const kernel_impl_params& impl_param);

std::string histc_inst::to_string(const histc_node& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    json_composite histc_info;
    histc_info.add("input id", node.input().id());
    histc_info.add("bins", desc->bins);
    histc_info.add("min", desc->min);
    histc_info.add("max", desc->max);
    node_info->add("histc info", histc_info);

    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

}  // namespace cldnn
