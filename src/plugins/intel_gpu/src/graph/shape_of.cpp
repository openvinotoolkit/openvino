// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shape_of_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include "to_string_utils.h"
#include <string>
#include <vector>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(shape_of)

layout shape_of_inst::calc_output_layout(shape_of_node const& node, kernel_impl_params const& impl_param) {
    auto prim = impl_param.typed_desc<shape_of>();

    data_types dt = data_types::i32;

    if (prim->output_data_types[0])
        dt = *prim->output_data_types[0];

    if (impl_param.has_fused_primitives()) {
        dt = impl_param.get_fused_output_layout().data_type;
    }

    cldnn::tensor out_size{static_cast<tensor::value_type>(prim->input_rank), 1, 1, 1};

    return layout{dt, format::bfyx, out_size};
}

template<typename ShapeType>
std::vector<layout> shape_of_inst::calc_output_layouts(shape_of_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto prim = impl_param.typed_desc<shape_of>();

    data_types output_dt = prim->output_data_types[0].value_or(data_types::i32);
    if (impl_param.has_fused_primitives()) {
        output_dt = impl_param.get_fused_output_layout().data_type;
    }

    auto in_shape = impl_param.get_input_layout(0).get<ShapeType>();
    auto output_shape = ShapeType{static_cast<int64_t>(in_shape.size())};

    return { layout{output_shape, output_dt, format::bfyx} };
}

template std::vector<layout> shape_of_inst::calc_output_layouts<ov::PartialShape>(shape_of_node const& node, const kernel_impl_params& impl_param);

std::string shape_of_inst::to_string(shape_of_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream primitive_description;

    json_composite shape_of_info;
    if (desc->output_data_types[0].has_value())
        shape_of_info.add("out dt: ", dt_to_str(*desc->output_data_types[0]));
    node_info->add("shape_of info", shape_of_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

shape_of_inst::typed_primitive_inst(network& network, shape_of_node const& node) : parent(network, node, true) { }
}  // namespace cldnn
