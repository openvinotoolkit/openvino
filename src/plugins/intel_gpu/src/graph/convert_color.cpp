// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_color_inst.h"

#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(convert_color)

layout convert_color_inst::calc_output_layout(convert_color_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<convert_color>();
    return desc->output_layout;
}

std::string convert_color_inst::to_string(convert_color_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite convert_color_info;
    convert_color_info.add("input id", input.id());
    convert_color_info.add("memory type", desc->mem_type);
    convert_color_info.add("input color format", desc->input_color_format);
    convert_color_info.add("output color format", desc->output_color_format);

    node_info->add("convert_color info", convert_color_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

convert_color_inst::typed_primitive_inst(network& network, convert_color_node const& node) : parent(network, node) {}

}  // namespace cldnn
