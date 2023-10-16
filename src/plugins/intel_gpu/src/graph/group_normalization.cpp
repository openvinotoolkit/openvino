// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "group_normalization_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(group_normalization)

layout group_normalization_inst::calc_output_layout(group_normalization_node const& node, kernel_impl_params const& impl_param) {
    assert(static_cast<bool>(impl_param.desc->output_data_types[0]) == false &&
        "Output data type forcing is not supported for group_normalization_node!");
    auto output_layout = impl_param.get_input_layout();

    if (impl_param.has_fused_primitives())
        output_layout.data_type = impl_param.get_fused_output_layout().data_type;

    return output_layout;
}

std::string group_normalization_inst::to_string(group_normalization_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite group_normalization_info;
    group_normalization_info.add("dimension", desc->num_groups);
    group_normalization_info.add("epsilon", desc->epsilon);

    node_info->add("group_normalization_info", group_normalization_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

group_normalization_inst::typed_primitive_inst(network& network, group_normalization_node const& node) : parent(network, node) {
}

} // namespace cldnn
