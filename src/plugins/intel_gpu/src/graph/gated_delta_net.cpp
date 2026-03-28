// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "gated_delta_net_inst.h"
#include "gated_delta_net_shape_inference.hpp"
#include "json_object.h"
#include "openvino/op/gated_delta_net.hpp"
#include "primitive_type_base.h"
#include "to_string_utils.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(gated_delta_net)

layout gated_delta_net_inst::calc_output_layout(const gated_delta_net_node& node, const kernel_impl_params& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template <typename ShapeType>
std::vector<layout> gated_delta_net_inst::calc_output_layouts(const gated_delta_net_node& node, const kernel_impl_params& impl_param) {
    const auto& desc = impl_param.typed_desc<gated_delta_net>();
    const auto& all_inputs = node.get_input_layouts();
    const auto num_outputs = desc->output_size();
    OPENVINO_ASSERT(all_inputs.size() == 6, "gated_delta_net's must have 6 inputs");

    ov::op::internal::GatedDeltaNet op;

    std::vector<ShapeType> input_shapes = {
        impl_param.get_input_layout(0).get<ShapeType>(),
        impl_param.get_input_layout(1).get<ShapeType>(),
        impl_param.get_input_layout(2).get<ShapeType>(),
        impl_param.get_input_layout(3).get<ShapeType>(),
        impl_param.get_input_layout(4).get<ShapeType>(),
        impl_param.get_input_layout(5).get<ShapeType>(),
    };
    auto output_shapes = ov::op::internal::shape_infer(&op, input_shapes);
    OPENVINO_ASSERT(num_outputs <= output_shapes.size(),
                    "gated_delta_net shape infer produced fewer outputs than requested: ",
                    output_shapes.size(),
                    " < ",
                    num_outputs);

    const auto value_layout = impl_param.get_input_layout(2);
    const auto state_layout = impl_param.get_input_layout(3);
    std::vector<layout> output_layouts;
    output_layouts.emplace_back(output_shapes[0], value_layout.data_type, value_layout.format);
    if (num_outputs == 2) {
        output_layouts.emplace_back(output_shapes[1], state_layout.data_type, state_layout.format);
    }
    return output_layouts;
}

template std::vector<layout> gated_delta_net_inst::calc_output_layouts<ov::PartialShape>(const gated_delta_net_node& node,
                                                                                         const kernel_impl_params& impl_param);

std::string gated_delta_net_inst::to_string(const gated_delta_net_node& node) {
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

gated_delta_net_inst::typed_primitive_inst(network& network, const gated_delta_net_node& node) : parent(network, node) {}
}  // namespace cldnn
