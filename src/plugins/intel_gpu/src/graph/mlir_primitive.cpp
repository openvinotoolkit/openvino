// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_primitive_inst.h"

#include <sstream>
#include <string>

#include "json_object.h"
#include "openvino/core/partial_shape.hpp"
#include "primitive_type_base.h"

namespace cldnn {

GPU_DEFINE_PRIMITIVE_TYPE_ID(mlir_primitive)

layout mlir_primitive_inst::calc_output_layout(const mlir_primitive_node& node,
                                               const kernel_impl_params& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template <typename ShapeType>
std::vector<layout> mlir_primitive_inst::calc_output_layouts(mlir_primitive_node const& /*node*/,
                                                             const kernel_impl_params& impl_param) {
    auto prim = impl_param.typed_desc<mlir_primitive>();

    std::vector<ShapeType> input_shapes;
    input_shapes.reserve(impl_param.input_layouts.size());
    for (const auto& l : impl_param.input_layouts) {
        input_shapes.push_back(l.get<ShapeType>());
    }

    std::vector<ov::PartialShape> output_shapes = prim->shape_infer_f(input_shapes);

    std::vector<layout> out_layouts;
    out_layouts.reserve(output_shapes.size());
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        out_layouts.emplace_back(output_shapes[i],
                                 prim->get_output_data_type(i).value(),
                                 format::get_default_format(output_shapes[i].size()));
    }
    return out_layouts;
}

template std::vector<layout> mlir_primitive_inst::calc_output_layouts<ov::PartialShape>(
        mlir_primitive_node const&, const kernel_impl_params&);

std::string mlir_primitive_inst::to_string(mlir_primitive_node const& node) {
    auto node_info = node.desc_to_json();
    std::stringstream primitive_description;
    json_composite mlir_info;
    if (const auto& op = node.get_primitive()->op) {
        mlir_info.add("subgraph_name", op->get_friendly_name());
    }
    mlir_info.add("num_outputs", std::to_string(node.get_primitive()->num_outputs));
    node_info->add("mlir_primitive_info", mlir_info);
    node_info->dump(primitive_description);
    return primitive_description.str();
}

typed_primitive_inst<mlir_primitive>::typed_primitive_inst(network& network,
                                                           const mlir_primitive_node& node)
    : parent(network, node),
      node(&node) {}

}  // namespace cldnn
