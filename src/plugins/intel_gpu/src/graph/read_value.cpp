// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <read_value_inst.h>
#include "primitive_type_base.h"
#include <sstream>
#include <json_object.h>

#include "read_value_shape_inference.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(read_value)

read_value_inst::typed_primitive_inst(network& network, const read_value_node& node) :
    parent(network, node, false),
    memory_state::variable{node.get_primitive()->variable_id} {
}

layout read_value_inst::calc_output_layout(const read_value_node& node, kernel_impl_params const& impl_param) {
    return impl_param.typed_desc<read_value>()->output_layout;
}

template<typename ShapeType>
std::vector<layout> read_value_inst::calc_output_layouts(const read_value_node& /*node*/, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<read_value>();
    auto input_layout = impl_param.get_input_layout(0);
    auto dt = desc->output_data_types[0].value_or(input_layout.data_type);

    ov::op::v6::ReadValue op;

    std::vector<ShapeType> output_shapes = { ShapeType{} };
    std::vector<ShapeType> input_shapes = {
        input_layout.get<ShapeType>()
    };

    ov::op::v6::shape_infer(&op, input_shapes, output_shapes);
    return {{output_shapes[0], dt, format::get_default_format(output_shapes[0].size())}};
}

std::string read_value_inst::to_string(const read_value_node& node) {
    auto node_info = node.desc_to_json();

    json_composite read_value_info;
    read_value_info.add("input id", node.input().id());
    read_value_info.add("variable id", node.get_primitive()->variable_id);
    node_info->add("read_value info", read_value_info);

    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

void read_value_inst::save(cldnn::BinaryOutputBuffer& ob) const {
    parent::save(ob);

    ob << variable_id();
}

void read_value_inst::load(cldnn::BinaryInputBuffer& ib) {
    parent::load(ib);

    std::string variable_id;
    ib >> variable_id;
    set_variable_id(variable_id);
}
} // namespace cldnn
