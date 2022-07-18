// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sstream>

#include "json_object.h"
#include "primitive_type_base.h"
#include "roll_inst.hpp"

namespace cldnn {

primitive_type_id roll::type_id() {
    static primitive_type_base<roll> instance;
    return &instance;
}

layout roll_inst::calc_output_layout(const roll_node& node) {
    return node.input().get_output_layout();
}

std::string roll_inst::to_string(const roll_node& node) {
    auto node_info = node.desc_to_json();
    json_composite roll_info;
    roll_info.add("input id", node.input().id());
    roll_info.add("shift", node.get_primitive()->shift);
    node_info->add("roll info", roll_info);
    std::ostringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

}  // namespace cldnn
