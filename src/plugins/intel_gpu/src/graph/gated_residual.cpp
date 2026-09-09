// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "gated_residual_inst.h"
#include "json_object.h"
#include "primitive_type_base.h"
#include "to_string_utils.h"

namespace cldnn {

GPU_DEFINE_PRIMITIVE_TYPE_ID(gated_residual)

layout gated_residual_inst::calc_output_layout(const gated_residual_node& node, const kernel_impl_params& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template <typename ShapeType>
std::vector<layout> gated_residual_inst::calc_output_layouts(const gated_residual_node& /*node*/, const kernel_impl_params& impl_param) {
    const auto& desc = impl_param.typed_desc<gated_residual>();
    const auto residual_layout = impl_param.get_input_layout(0);
    const auto out_dt = desc->output_data_types.empty() || !desc->output_data_types[0].has_value() ? residual_layout.data_type
                                                                                                  : desc->output_data_types[0].value();

    if (desc->mode == gated_residual::Mode::WRITE) {
        // R' keeps the widened residual layout.
        return {layout{residual_layout.get<ShapeType>(), out_dt, residual_layout.format}};
    }

    auto mixed_shape = residual_layout.get<ShapeType>();
    OPENVINO_ASSERT(mixed_shape.rank().is_static() && mixed_shape.rank().get_length() >= 2,
                    "[GPU] gated_residual requires a static-rank residual input of rank >= 2");
    mixed_shape[mixed_shape.rank().get_length() - 1] = ov::Dimension(static_cast<int64_t>(desc->hidden_size));

    std::vector<layout> output_layouts;
    output_layouts.emplace_back(mixed_shape, out_dt, residual_layout.format);

    if (desc->produce_write_gate) {
        auto gate_shape = residual_layout.get<ShapeType>();
        gate_shape[gate_shape.rank().get_length() - 1] = ov::Dimension(static_cast<int64_t>(desc->hc_count));
        const auto gate_dt = desc->output_data_types.size() > 1 && desc->output_data_types[1].has_value() ? desc->output_data_types[1].value() : out_dt;
        output_layouts.emplace_back(gate_shape, gate_dt, residual_layout.format);
    }

    return output_layouts;
}

template std::vector<layout> gated_residual_inst::calc_output_layouts<ov::PartialShape>(const gated_residual_node& node,
                                                                                       const kernel_impl_params& impl_param);

std::string gated_residual_inst::to_string(const gated_residual_node& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream primitive_description;

    json_composite info;
    info.add("mode", static_cast<int>(desc->mode));
    info.add("hc_count", desc->hc_count);
    info.add("hidden_size", desc->hidden_size);
    info.add("lowrank", desc->lowrank);
    info.add("epsilon", desc->epsilon);
    info.add("norm_gain_plus_one", desc->norm_gain_plus_one);
    info.add("produce_write_gate", desc->produce_write_gate);

    node_info->add("gated_residual_info", info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

gated_residual_inst::typed_primitive_inst(network& network, const gated_residual_node& node) : parent(network, node) {}

}  // namespace cldnn
