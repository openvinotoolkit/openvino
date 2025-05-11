// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rms_inst.h"

#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(rms);

layout rms_inst::calc_output_layout(rms_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<rms>();
    auto input_layout = impl_param.get_input_layout();
    auto output_type = desc->output_data_types[0].value_or(input_layout.data_type);
    auto output_format = input_layout.format;

    return layout(output_type, output_format, input_layout.get_tensor());
}

std::string rms_inst::to_string(rms_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite rms_info;
    rms_info.add("input_id", node.input(0).id());
    rms_info.add("epsilon", desc->epsilon);

    node_info->add("rms_info", rms_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

rms_inst::typed_primitive_inst(network& network, rms_node const& node) : parent(network, node) {}

}  // namespace cldnn
