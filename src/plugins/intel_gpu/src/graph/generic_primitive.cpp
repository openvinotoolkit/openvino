// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "generic_primitive_inst.h"
#include "openvino/core/partial_shape.hpp"
#include "primitive_type_base.h"
#include <sstream>
#include "json_object.h"
#include <string>

namespace cldnn {

primitive_type_id generic_primitive::type_id() {
    static primitive_type_base<generic_primitive> instance;
    return &instance;
}

layout generic_primitive_inst::calc_output_layout(const generic_primitive_node& node, const kernel_impl_params& impl_param) {
   return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template<typename ShapeType>
std::vector<layout> generic_primitive_inst::calc_output_layouts(generic_primitive_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto prim = impl_param.typed_desc<generic_primitive>();

    std::vector<ShapeType> input_shapes;
    for (const auto& l : impl_param.input_layouts) {
        input_shapes.push_back(l.get<ShapeType>());
    }

    std::vector<ShapeType> output_shapes = prim->shape_infer_f(input_shapes);

    std::vector<layout> out_layouts;
    for (size_t i = 0; i < output_shapes.size(); i++) {
        out_layouts.emplace_back(output_shapes[i], prim->get_output_data_type(i).value(), format::get_default_format(output_shapes[i].size()));
    }

    return out_layouts;
}

std::string generic_primitive_inst::to_string(generic_primitive_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite generic_prim_info;
    node_info->add("custom primitive info", generic_prim_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

generic_primitive_inst::typed_primitive_inst(network& network, generic_primitive_node const& node) : parent(network, node) {}

}  // namespace cldnn
