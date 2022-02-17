// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scatter_nd_update_inst.h"

#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id scatter_nd_update::type_id() {
    static primitive_type_base<scatter_nd_update> instance;
    return &instance;
}


layout scatter_nd_update_inst::calc_output_layout(scatter_nd_update_node const& node) {
    auto input_layout = node.input(0).get_output_layout();

    auto output_shape = input_layout.size;
    auto input_format = input_layout.format;
    auto output_type = input_layout.data_type;

    if (node.has_fused_primitives()) {
        output_type = node.get_fused_output_layout().data_type;
    }

    return layout{output_type, input_format, output_shape};
}

std::string scatter_nd_update_inst::to_string(scatter_nd_update_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite scatter_nd_update_info;
    scatter_nd_update_info.add("input id", input.id());
    scatter_nd_update_info.add("input shape", node.input(0).get_output_layout().size.to_string());
    scatter_nd_update_info.add("indices shape", node.input(1).get_output_layout().size.to_string());
    scatter_nd_update_info.add("updates shape", node.input(2).get_output_layout().size.to_string());

    node_info->add("scatter_nd_update info", scatter_nd_update_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

scatter_nd_update_inst::typed_primitive_inst(network& network, scatter_nd_update_node const& node) : parent(network, node) {}

}  // namespace cldnn
