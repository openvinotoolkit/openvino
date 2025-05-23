// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <json_object.h>
#include <search_sorted_inst.h>

#include <sstream>

#include "openvino/core/enum_names.hpp"
#include "primitive_type_base.h"
#include "search_sorted_shape_inference.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(search_sorted)

search_sorted_inst::typed_primitive_inst(network& network, search_sorted_node const& node) : parent(network, node) {}

layout search_sorted_inst::calc_output_layout(search_sorted_node const& node, kernel_impl_params const& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template <typename ShapeType>
std::vector<layout> search_sorted_inst::calc_output_layouts(search_sorted_node const& node,
                                                            kernel_impl_params const& impl_param) {
    auto primitive = impl_param.typed_desc<search_sorted>();

    auto input0_layout = impl_param.get_input_layout(0);
    auto input1_layout = impl_param.get_input_layout(1);

    const data_types output_type = impl_param.desc->output_data_types[0].value_or(data_types::i64);

    std::vector<ShapeType> input_shapes = {
        input0_layout.get<ShapeType>(),  // sorted shape
        input1_layout.get<ShapeType>(),  // values shape
    };

    std::vector<ShapeType> output_shapes;

    ov::op::v15::SearchSorted op;
    op.set_right_mode(primitive->right_mode);
    output_shapes = shape_infer(&op, input_shapes);

    return {layout{output_shapes[0], output_type, input1_layout.format}};
}

std::string search_sorted_inst::to_string(search_sorted_node const& node) {
    auto node_info = node.desc_to_json();
    json_composite search_sorted_info;
    search_sorted_info.add("sorted id", node.input(0).id());
    search_sorted_info.add("values id", node.input(1).id());
    search_sorted_info.add("right_mode", node.get_primitive()->right_mode);
    node_info->add("search_sorted info", search_sorted_info);
    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

}  // namespace cldnn