// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>

#include "fused_mlp_inst.h"
#include "json_object.h"
#include "primitive_type_base.h"
#include "program_node.h"

namespace cldnn {

GPU_DEFINE_PRIMITIVE_TYPE_ID(fused_mlp)

layout fused_mlp_inst::calc_output_layout(const fused_mlp_node& /*node*/, const kernel_impl_params& impl_param) {
    return impl_param.input_layouts[0];
}

template <typename ShapeType>
std::vector<layout> fused_mlp_inst::calc_output_layouts(const fused_mlp_node& /*node*/, const kernel_impl_params& impl_param) {
    return {impl_param.input_layouts[0]};
}

template std::vector<layout> fused_mlp_inst::calc_output_layouts<ov::PartialShape>(const fused_mlp_node& node, const kernel_impl_params& impl_param);

std::string fused_mlp_inst::to_string(const fused_mlp_node& node) {
    auto node_info = node.desc_to_json();
    json_composite fused_mlp_info;
    node_info->add("fused_mlp info", fused_mlp_info);
    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

fused_mlp_inst::typed_primitive_inst(network& network, const fused_mlp_node& node) : parent(network, node) {}

}  // namespace cldnn

