// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "linear_attention_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include "to_string_utils.h"
#include <string>
#include <vector>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(linear_attention)

layout linear_attention_inst::calc_output_layout(linear_attention_node const& node, kernel_impl_params const& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template<typename ShapeType>
std::vector<layout> linear_attention_inst::calc_output_layouts(linear_attention_node const& node, const kernel_impl_params& impl_param) {
    const auto& desc = impl_param.typed_desc<linear_attention>();
    const auto& all_inputs = node.get_input_layouts();
    const auto num_outputs = desc->output_size();
    if (all_inputs.size() != 6)
        OPENVINO_THROW("linear_attention's must have 6 inputs");
    // query, key, value, g, beta, initial_states
    auto query_layout = impl_param.get_input_layout(0);
    auto value_layout = impl_param.get_input_layout(2);
    auto out_ps = value_layout.get_partial_shape();
    const auto& q_ps = query_layout.get_partial_shape();
    if (out_ps.rank().is_static() && q_ps.rank().is_static() && out_ps.rank().get_length() == 4 && q_ps.rank().get_length() == 4) {
        out_ps[0] = q_ps[0];
        out_ps[1] = q_ps[1];
    }
    std::vector<layout> output_layouts;
    output_layouts.emplace_back(out_ps, value_layout.data_type, value_layout.format);
    if (num_outputs == 2) {
        output_layouts.push_back(impl_param.get_input_layout(5));
    }
    return output_layouts;
}

template std::vector<layout> linear_attention_inst::calc_output_layouts<ov::PartialShape>(linear_attention_node const& node, const kernel_impl_params& impl_param);

std::string linear_attention_inst::to_string(linear_attention_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream primitive_description;

    json_composite linear_attention_info;
    linear_attention_info.add("query", node.input(0).id());
    linear_attention_info.add("key", node.input(1).id());
    linear_attention_info.add("value", node.input(2).id());
    linear_attention_info.add("g", node.input(3).id());
    linear_attention_info.add("beta", node.input(3).id());

    node_info->add("linear_attention_info", linear_attention_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

linear_attention_inst::typed_primitive_inst(network& network, linear_attention_node const& node) : parent(network, node) { }
}  // namespace cldnn
