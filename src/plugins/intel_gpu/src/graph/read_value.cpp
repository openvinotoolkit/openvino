// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <read_value_inst.h>
#include "primitive_type_base.h"
#include <sstream>
#include <json_object.h>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(read_value)

read_value_inst::typed_primitive_inst(network& network, const read_value_node& node) :
    parent(network, node, !node.can_be_optimized() && (node.get_output_layout().is_static() || node.get_output_layout().has_upper_bound())),
    memory_state::variable{node.get_primitive()->variable_id} {
}

layout read_value_inst::calc_output_layout(const read_value_node& node, kernel_impl_params const& impl_param) {
    return impl_param.typed_desc<read_value>()->output_layout;
}

std::string read_value_inst::to_string(const read_value_node& node) {
    auto node_info = node.desc_to_json();

    json_composite read_value_info;
    read_value_info.add("variable id", node.get_primitive()->variable_id);
    node_info->add("read_value info", read_value_info);

    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

void read_value_inst::on_execute() {
    update_output_memory();
}

void read_value_inst::update_output_memory() {
    if (!can_be_optimized() || !get_network().has_variable(variable_id()))
        return;

    const auto& variable = get_network().get_variable(variable_id());
    set_output_memory(variable.get_memory(), false, 0);
}
} // namespace cldnn
