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

kernel_impl_params custom_gpu_primitive_inst::get_fake_aligned_params(kernel_impl_params const& orig_impl_param) {
    auto updated_param = std::move(orig_impl_param);
    const auto& orig_input_layout = orig_impl_param.get_input_layout();
    const auto& orig_output_layout = orig_impl_param.get_output_layout();
    OPENVINO_ASSERT(orig_input_layout.is_static() && orig_output_layout.is_static(),
                    "in/out layouts should be static for fake alignment!");

    auto output_shape = orig_output_layout.get_partial_shape().to_shape();

    // auto op = std::static_pointer_cast<const custom_gpu_primitive>(updated_param.desc);
    updated_param.custom_op_dynamic_gws = output_shape;

    custom_gpu_primitive::update_work_group_size(std::shared_ptr<ov::Node>());

    // updated_param.output_layouts[0] = orig_output_layout.clone_with_other_shape(output_shape);
    // std::cout << "Apply fake alignment: input(" << orig_input_layout.to_short_string() << " -> "
    //                        << updated_param.input_layouts[0].to_short_string() << "), output(" << orig_output_layout.to_short_string() << " -> "
    //                        << updated_param.output_layouts[0].to_short_string() << ")\n";

    return updated_param;
}

custom_gpu_primitive_inst::typed_primitive_inst(network& network, custom_gpu_primitive_node const& node)
    : parent(network, node) {}
}  // namespace cldnn
