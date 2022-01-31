// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "edgpsi_inst.hpp"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id edgpsi::type_id() {
    static primitive_type_base<edgpsi> instance;
    return &instance;
}

layout edgpsi_inst::calc_output_layout(const edgpsi_node& node) {
    const layout data_layout = node.input().get_output_layout();
    auto desc = node.get_primitive();

    return layout(data_layout.data_type, format::bfyx, {static_cast<int>(desc->post_nms_count), 4, 1, 1});
}

std::string edgpsi_inst::to_string(const edgpsi_node& node) {
    auto desc = node.get_primitive();

    std::stringstream primitive_description;

    json_composite ed_info;
    ed_info.add("min_size", desc->min_size);
    ed_info.add("nms_threshold", desc->nms_threshold);
    ed_info.add("pre_nms_count", desc->pre_nms_count);
    ed_info.add("post_nms_count", desc->post_nms_count);

    auto node_info = node.desc_to_json();
    node_info->add("edgpsi_info", ed_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}
}  // namespace cldnn
