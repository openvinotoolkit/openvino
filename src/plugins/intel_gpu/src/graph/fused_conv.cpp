// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fused_conv_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include "to_string_utils.h"
#include <string>
#include <vector>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(fused_conv)

layout fused_conv_inst::calc_output_layout(fused_conv_node const& node, kernel_impl_params const& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template<typename ShapeType>
std::vector<layout> fused_conv_inst::calc_output_layouts(fused_conv_node const& node, const kernel_impl_params& impl_param) {
    const auto& desc = impl_param.typed_desc<fused_conv>();
    const auto num_outputs = desc->output_size();
    if (impl_param.input_layouts.size() != 4)
        OPENVINO_THROW("fused_conv must have 4 inputs");

    // input[0]: [B, conv_dim, S] -> output[0]: [B, conv_dim, S]
    auto input_layout = impl_param.get_input_layout(0);
    // input[3]: [B, conv_dim, kernel_size] -> output[1]: [B, conv_dim, kernel_size]
    auto state_layout = impl_param.get_input_layout(3);

    std::vector<layout> output_layouts;
    output_layouts.emplace_back(input_layout.get_partial_shape(), input_layout.data_type, input_layout.format);
    if (num_outputs == 2) {
        output_layouts.push_back(state_layout);
    }
    return output_layouts;
}

template std::vector<layout> fused_conv_inst::calc_output_layouts<ov::PartialShape>(fused_conv_node const& node, const kernel_impl_params& impl_param);

std::string fused_conv_inst::to_string(fused_conv_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream primitive_description;

    json_composite fused_conv_info;
    fused_conv_info.add("input", node.input(0).id());
    fused_conv_info.add("conv_weight", node.input(1).id());
    fused_conv_info.add("beam_idx", node.input(2).id());
    fused_conv_info.add("initial_state", node.input(3).id());

    node_info->add("fused_conv_info", fused_conv_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

fused_conv_inst::typed_primitive_inst(network& network, fused_conv_node const& node) : parent(network, node) { }
}  // namespace cldnn
