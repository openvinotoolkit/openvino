// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gated_delta_net_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include "to_string_utils.h"
#include <string>
#include <vector>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(gated_delta_net)

layout gated_delta_net_inst::calc_output_layout(gated_delta_net_node const& node, kernel_impl_params const& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template<typename ShapeType>
std::vector<layout> gated_delta_net_inst::calc_output_layouts(gated_delta_net_node const& node, const kernel_impl_params& impl_param) {
    const auto& desc = impl_param.typed_desc<gated_delta_net>();
    const auto& all_inputs = node.get_input_layouts();
    if (all_inputs.size() != 6)
        OPENVINO_THROW("gated_delta_net's must have 6 inputs");
    // query, key, value, g, beta, initial_states
    auto value_layout = impl_param.get_input_layout(2);
    auto states_layout = impl_param.get_input_layout(5);
    auto output_layout = layout{value_layout.get_partial_shape(), value_layout.data_type, value_layout.format};
    return {output_layout, states_layout};
}

template std::vector<layout> gated_delta_net_inst::calc_output_layouts<ov::PartialShape>(gated_delta_net_node const& node, const kernel_impl_params& impl_param);

std::string gated_delta_net_inst::to_string(gated_delta_net_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream primitive_description;

    json_composite gated_delta_net_info;
    gated_delta_net_info.add("query", node.input(0).id());
    gated_delta_net_info.add("key", node.input(1).id());
    gated_delta_net_info.add("value", node.input(2).id());
    gated_delta_net_info.add("recurrent_state", node.input(3).id());
    gated_delta_net_info.add("gate", node.input(4).id());
    gated_delta_net_info.add("beta", node.input(5).id());

    node_info->add("gated_delta_net_info", gated_delta_net_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

gated_delta_net_inst::typed_primitive_inst(network& network, gated_delta_net_node const& node) : parent(network, node) { }
}  // namespace cldnn
