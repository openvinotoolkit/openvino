// Copyright (C) 2018-2025 Intel Corporation
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

namespace {
data_types get_output_data_type(const kernel_impl_params& impl_param) {
    if (impl_param.has_fused_primitives()) {
        return impl_param.get_output_element_type();
    }

    const auto prim = impl_param.typed_desc<shape_of>();
    return prim->output_data_types[0].value_or(data_types::i32);
}
}  // namespace

layout shape_of_inst::calc_output_layout(shape_of_node const& node, kernel_impl_params const& impl_param) {
    const auto prim = impl_param.typed_desc<shape_of>();
    const auto dt = get_output_data_type(impl_param);
    const auto rank = impl_param.get_input_layout(0).get_rank();
    const cldnn::tensor out_size{static_cast<tensor::value_type>(rank), 1, 1, 1};

    return layout{dt, format::bfyx, out_size};
}

template<typename ShapeType>
std::vector<layout> shape_of_inst::calc_output_layouts(shape_of_node const& /*node*/, const kernel_impl_params& impl_param) {
    const auto dt = get_output_data_type(impl_param);
    const auto in_shape = impl_param.get_input_layout(0).get<ShapeType>();
    const auto output_shape = ShapeType{static_cast<int64_t>(in_shape.size())};

    return { layout{output_shape, dt, format::bfyx} };
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
