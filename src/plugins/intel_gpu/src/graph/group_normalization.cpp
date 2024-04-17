// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "group_normalization_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(group_normalization)

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
