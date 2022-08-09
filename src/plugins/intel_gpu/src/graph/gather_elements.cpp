// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_elements_inst.h"

#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id gather_elements::type_id() {
    static primitive_type_base<gather_elements> instance;
    return &instance;
}

layout gather_elements_inst::calc_output_layout(gather_elements_node const& node, kernel_impl_params const& impl_param) {
    auto op = impl_param.typed_desc<gather_elements>();

    auto input_layout_origin = impl_param.get_input_layout(0);
    auto indices_layout_origin = impl_param.get_input_layout(1);

    auto input_layout = input_layout_origin.get_tensor().sizes(input_layout_origin.format);
    auto indices_layout = indices_layout_origin.get_tensor().sizes(indices_layout_origin.format);

    auto output_type = (impl_param.has_fused_primitives()) ? impl_param.get_fused_output_layout().data_type :
                       input_layout_origin.data_type;
    auto output_shape = op->output_shape;
    auto output_format = op->output_format;
    // calculate initial output shape
    return layout(output_type, output_format, output_shape);
}

std::string gather_elements_inst::to_string(gather_elements_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite gather_elements_info;
    gather_elements_info.add("input id", input.id());
    gather_elements_info.add("output format", calc_output_layout(node, *node.get_kernel_impl_params()).format);
    gather_elements_info.add("axis", desc->axis);

    node_info->add("gather_elements info", gather_elements_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

gather_elements_inst::typed_primitive_inst(network& network, gather_elements_node const& node) : parent(network, node) {}

}  // namespace cldnn
