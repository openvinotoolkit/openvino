// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gated_mlp_inst.h"

#include "json_object.h"
#include "matmul_shape_inference.hpp"
#include "primitive_type_base.h"

#include <string>

namespace cldnn {

GPU_DEFINE_PRIMITIVE_TYPE_ID(gated_mlp)

layout gated_mlp_inst::calc_output_layout(gated_mlp_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<gated_mlp>();
    auto input_layout = impl_param.get_input_layout();
    auto output_type = impl_param.desc->output_data_types[0].value_or(input_layout.data_type);
    auto output_format = input_layout.format;

    return layout(output_type, output_format, desc->output_size);
}

template <typename ShapeType>
std::vector<layout> gated_mlp_inst::calc_output_layouts(gated_mlp_node const& node, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<gated_mlp>();
    auto input_layout = impl_param.get_input_layout();
    auto output_type = impl_param.desc->output_data_types[0].value_or(input_layout.data_type);
    auto output_format = input_layout.format;

    std::vector<ShapeType> input_shapes = {
        impl_param.get_input_layout(0).get<ShapeType>(),
        impl_param.get_input_layout(1).get<ShapeType>(),
        impl_param.get_input_layout(2).get<ShapeType>(),
        impl_param.get_input_layout(3).get<ShapeType>()
    };

    ov::op::v0::MatMul matmul;
    matmul.set_transpose_a(false);
    matmul.set_transpose_b(false);

    auto up_shapes = ov::op::v0::shape_infer(&matmul, std::vector<ShapeType>{input_shapes[0], input_shapes[2]});
    auto gate_shapes = ov::op::v0::shape_infer(&matmul, std::vector<ShapeType>{input_shapes[0], input_shapes[1]});

    OPENVINO_ASSERT(up_shapes[0].compatible(gate_shapes[0]),
                    "GatedMLP requires gate/up projection output shapes to match.");

    auto out_shapes = ov::op::v0::shape_infer(&matmul, std::vector<ShapeType>{up_shapes[0], input_shapes[3]});

    return {layout(out_shapes[0], output_type, output_format)};
}

template std::vector<layout> gated_mlp_inst::calc_output_layouts<ov::PartialShape>(gated_mlp_node const& node,
                                                                                    const kernel_impl_params& impl_param);

std::string gated_mlp_inst::to_string(gated_mlp_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;
    json_composite gated_mlp_info;
    gated_mlp_info.add("input_id", node.input().id());
    gated_mlp_info.add("weights_gate_id", node.weights_gate().id());
    gated_mlp_info.add("weights_up_id", node.weights_up().id());
    gated_mlp_info.add("weights_down_id", node.weights_down().id());
    gated_mlp_info.add("activation", static_cast<int64_t>(desc->activation));

    node_info->add("gated_mlp_info", gated_mlp_info);
    node_info->dump(primitive_description);
    return primitive_description.str();
}

gated_mlp_inst::typed_primitive_inst(network& network, gated_mlp_node const& node) : parent(network, node) {}

}  // namespace cldnn
