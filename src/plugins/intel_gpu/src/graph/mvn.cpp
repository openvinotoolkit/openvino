// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(mvn)

std::string mvn_inst::to_string(mvn_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();
    auto epsilon = desc->epsilon;
    auto axes = desc->reduction_axes;
    auto normalize_variance = desc->normalize_variance ? "true" : "false";
    auto eps_inside_sqrt = desc->eps_inside_sqrt ? "true" : "false";
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite mvn_info;
    mvn_info.add("input id", input.id());
    mvn_info.add("epsilon", epsilon);
    mvn_info.add("reduction axes", std::move(axes));
    mvn_info.add("normalize_variance region", normalize_variance);
    mvn_info.add("eps_inside_sqrt region", eps_inside_sqrt);

    node_info->add("mvn info", mvn_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

mvn_inst::typed_primitive_inst(network& network, mvn_node const& node) : parent(network, node) {}
}  // namespace cldnn
