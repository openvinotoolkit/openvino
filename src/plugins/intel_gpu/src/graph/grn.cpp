// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "grn_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(grn)

layout grn_inst::calc_output_layout(grn_node const& node, kernel_impl_params const& impl_param) {
    auto input_node_layout = impl_param.get_non_padded_input_layout();
    auto output_type = impl_param.desc->output_data_types[0].value_or(input_node_layout.data_type);

    return layout(output_type, input_node_layout.format, input_node_layout.get_tensor());
}

std::string grn_inst::to_string(grn_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();
    auto bias = desc->bias;
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite grn_info;
    grn_info.add("input id", input.id());
    grn_info.add("bias", bias);

    node_info->add("grn info", grn_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

grn_inst::typed_primitive_inst(network& network, grn_node const& node) : parent(network, node) {}
}  // namespace cldnn
