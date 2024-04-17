// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rms_inst.h"

#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(rms);

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
