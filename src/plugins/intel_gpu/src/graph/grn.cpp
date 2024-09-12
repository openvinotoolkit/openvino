// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "grn_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(grn)

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
