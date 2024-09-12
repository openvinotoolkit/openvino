// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "select_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

#include "select_shape_inference.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(select)

template<typename ShapeType>
std::vector<layout> select_inst::calc_output_layouts(const select_node& /*node*/, const kernel_impl_params& impl_param) {
    auto input0_layout = impl_param.get_input_layout(0);
    auto input1_layout = impl_param.get_input_layout(1);
    auto input2_layout = impl_param.get_input_layout(2);

    auto desc = impl_param.typed_desc<select>();
    auto dt = desc->output_data_types[0].value_or(input1_layout.data_type);
    if (impl_param.has_fused_primitives()) {
        dt = impl_param.get_output_element_type();
    }

    ov::op::v1::Select op;
    op.set_auto_broadcast(desc->broadcast_spec);

    std::vector<ShapeType> input_shapes = {
        input0_layout.get<ShapeType>(),
        input1_layout.get<ShapeType>(),
        input2_layout.get<ShapeType>()
    };

    std::vector<ShapeType> output_shapes = ov::op::v1::shape_infer(&op, input_shapes);

    return {{output_shapes[0], dt, format::get_default_format(output_shapes[0].size())}};
}

template std::vector<layout> select_inst::calc_output_layouts<ov::PartialShape>(select_node const& node, const kernel_impl_params& impl_param);

std::string select_inst::to_string(select_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream primitive_description;

    json_composite select_info;
    for (size_t i = 0; i < node.get_inputs_count(); i++) {
        select_info.add("input_" + std::to_string(i), node.input(i).id());
    }

    node_info->add("select info", select_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

select_inst::typed_primitive_inst(network& network, select_node const& node) : parent(network, node) {
    auto& deps = node.get_dependencies();

    auto dep0_out_layout = deps[0].first->get_output_layout();
    auto dep1_out_layout = deps[1].first->get_output_layout();
    auto dep2_out_layout = deps[2].first->get_output_layout();

    if (dep0_out_layout.is_dynamic() ||
        dep1_out_layout.is_dynamic() ||
        dep2_out_layout.is_dynamic())
        return;

    CLDNN_ERROR_LESS_THAN(node.id(),
                                "Number of inputs",
                                deps.size(),
                                "Expected number of inputs",
                                3,
                                "");
}
}  // namespace cldnn
