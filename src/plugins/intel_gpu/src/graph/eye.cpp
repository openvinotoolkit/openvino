// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <eye_inst.h>
#include <json_object.h>

#include <sstream>

#include "primitive_type_base.h"

namespace cldnn {

primitive_type_id eye::type_id() {
    static primitive_type_base<eye> instance;
    return &instance;
}

eye_inst::typed_primitive_inst(network& network, eye_node const& node) : parent(network, node) {}

layout eye_inst::calc_output_layout(eye_node const& node, const kernel_impl_params&) {
    auto primitive = node.get_primitive();
    return {*(primitive->output_data_type), node.input().get_output_layout().format, primitive->output_shape};
}

std::string eye_inst::to_string(eye_node const& node) {
    auto node_info = node.desc_to_json();
    json_composite eye_info;
    eye_info.add("rows id", node.get_dependency(0).id());
    eye_info.add("cols id", node.get_dependency(1).id());
    eye_info.add("diagInd id", node.get_dependency(2).id());
    if (node.get_dependencies().size() == 4)
        eye_info.add("batchShape id", node.get_dependency(3).id());
    node_info->add("slice info", eye_info);
    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

}  // namespace cldnn
