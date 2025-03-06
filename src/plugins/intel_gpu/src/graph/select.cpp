// Copyright (C) 2018-2025 Intel Corporation
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

layout select_inst::calc_output_layout(select_node const& node, kernel_impl_params const& impl_param) {
    assert(static_cast<bool>(impl_param.desc->output_data_types[0]) == false &&
           "Output data type forcing is not supported for select_node!");

    auto in_layout = impl_param.get_non_padded_input_layout(1);
    auto output_size = in_layout.get_tensor();

    if (impl_param.typed_desc<select>()->broadcast_spec.m_type == ov::op::AutoBroadcastType::NUMPY) {
        auto input1_size = impl_param.get_input_layout(1).get_tensor();
        auto input2_size = impl_param.get_input_layout(2).get_tensor();
        output_size = tensor::max(input1_size, input2_size);
        // Cond input0 also can be broadcasted.
        auto input0_size = impl_param.get_input_layout(0).get_tensor();
        output_size = tensor::max(input0_size, output_size);
    }

    return layout(in_layout.data_type, in_layout.format, output_size);
}

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

    bool allow_new_shape_infer = network.get_program()->get_config().get_allow_new_shape_infer();
    // Broadcast check is performed in ngraph shape infer of select when allow_new_shape_infer=true
    if (!allow_new_shape_infer) {
        if (node.get_primitive()->broadcast_spec.m_type == ov::op::AutoBroadcastType::NONE) {
            CLDNN_ERROR_LAYOUT_MISMATCH(node.id(),
                                    "Positive input layout",
                                    deps[1].first->get_output_layout(),
                                    "Negative input layout",
                                    deps[2].first->get_output_layout(),
                                    "");

            CLDNN_ERROR_NOT_EQUAL(node.id(),
                                    "Mask size",
                                    deps[0].first->get_output_layout().get_tensor(),
                                    "Positive input format",
                                    deps[1].first->get_output_layout().get_tensor(),
                                    "");
        } else if (node.get_primitive()->broadcast_spec.m_type == ov::op::AutoBroadcastType::NUMPY) {
            CLDNN_ERROR_DATA_TYPES_MISMATCH(node.id(),
                                    "Positive input data type",
                                    deps[1].first->get_output_layout().data_type,
                                    "Negative input data type",
                                    deps[2].first->get_output_layout().data_type,
                                    "");

            auto dep1_size = deps[1].first->get_output_layout().get_tensor();
            auto dep2_size = deps[2].first->get_output_layout().get_tensor();
            cldnn::tensor output_tensor = tensor::max(dep1_size, dep2_size);
            // Cond input0 also can be broadcasted.
            auto dep0_size = deps[0].first->get_output_layout().get_tensor();
            output_tensor = tensor::max(dep0_size, output_tensor);

            auto max_dim_count = output_tensor.raw.size();

            for (size_t i = 0; i < deps.size(); i++) {
                for (size_t d = 0; d < max_dim_count; d++) {
                    auto current_dim = deps[i].first->get_output_layout().get_tensor().raw[d];

                    CLDNN_ERROR_BOOL(node.id(),
                                        "Sizes equal or broadcast is possible",
                                        !(current_dim == output_tensor.raw[d] || current_dim == 1),
                                        "Invalid input shapes");
                }
            }
        } else {
            CLDNN_ERROR_MESSAGE(node.id(), "Unsupported broadcast_type: " + std::to_string(static_cast<int>(node.get_primitive()->broadcast_spec.m_type)));
        }
    }
}
}  // namespace cldnn
