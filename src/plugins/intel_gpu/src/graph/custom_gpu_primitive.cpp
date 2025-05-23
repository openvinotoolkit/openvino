// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "custom_gpu_primitive_inst.h"
#include "primitive_type_base.h"
#include <sstream>
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(custom_gpu_primitive)

std::string custom_gpu_primitive_inst::to_string(custom_gpu_primitive_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite custom_gpu_prim_info;
    custom_gpu_prim_info.add("entry point", desc->kernel_entry_point);
    custom_gpu_prim_info.add("kernels code", desc->kernels_code);
    custom_gpu_prim_info.add("build options", desc->build_options);
    custom_gpu_prim_info.add("gws", desc->gws);
    custom_gpu_prim_info.add("lws", desc->lws);
    // TODO: consider printing more information here
    node_info->add("custom primitive info", custom_gpu_prim_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

custom_gpu_primitive_inst::typed_primitive_inst(network& network, custom_gpu_primitive_node const& node)
    : parent(network, node) {}
}  // namespace cldnn
