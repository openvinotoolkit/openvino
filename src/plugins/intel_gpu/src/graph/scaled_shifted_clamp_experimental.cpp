// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scaled_shifted_clamp_experimental_inst.h"

#include <sstream>
#include <string>

#include "json_object.h"
#include "primitive_type_base.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(scaled_shifted_clamp_experimental);

layout scaled_shifted_clamp_experimental_inst::calc_output_layout(
    scaled_shifted_clamp_experimental_node const& /*node*/,
    kernel_impl_params const& impl_param) {
    const auto input_layout = impl_param.get_input_layout();
    const auto output_type  = impl_param.desc->output_data_types[0].value_or(input_layout.data_type);
    return layout(output_type, input_layout.format, input_layout.get_tensor());
}

template <typename ShapeType>
std::vector<layout> scaled_shifted_clamp_experimental_inst::calc_output_layouts(
    scaled_shifted_clamp_experimental_node const& /*node*/,
    const kernel_impl_params& impl_param) {
    const auto input_layout = impl_param.get_input_layout();
    const auto output_type  = impl_param.desc->output_data_types[0].value_or(input_layout.data_type);
    return {layout(input_layout.get<ShapeType>(), output_type, input_layout.format)};
}

template std::vector<layout> scaled_shifted_clamp_experimental_inst::calc_output_layouts<ov::PartialShape>(
    scaled_shifted_clamp_experimental_node const& node,
    const kernel_impl_params& impl_param);

std::string scaled_shifted_clamp_experimental_inst::to_string(scaled_shifted_clamp_experimental_node const& node) {
    const auto  desc      = node.get_primitive();
    auto        node_info = node.desc_to_json();
    const auto& input     = node.input();

    std::stringstream primitive_description;

    json_composite info;
    info.add("input_id", input.id());
    info.add("scale", desc->scale);
    info.add("bias", desc->bias);
    info.add("lo", desc->lo);
    info.add("hi", desc->hi);

    node_info->add("scaled_shifted_clamp_experimental_info", info);
    node_info->dump(primitive_description);
    return primitive_description.str();
}

scaled_shifted_clamp_experimental_inst::typed_primitive_inst(network& network,
                                                             scaled_shifted_clamp_experimental_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
