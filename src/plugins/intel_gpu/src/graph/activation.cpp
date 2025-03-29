// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "activation_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>
#include <vector>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(activation)

layout activation_inst::calc_output_layout(activation_node const& node, kernel_impl_params const& impl_param) {
    assert(static_cast<bool>(impl_param.desc->output_data_types[0]) == false &&
           "Output data type forcing is not supported for activation_node!");

    auto input_node_layout = impl_param.get_non_padded_input_layout();
    auto desc = impl_param.typed_desc<activation>();
    auto func = desc->activation_function;

    std::vector<activation_func> activations_int8 = {
        activation_func::none,
        activation_func::negative,
        activation_func::negation,
        activation_func::relu,
        activation_func::floor,
        activation_func::clamp,
        activation_func::abs };

    if (input_node_layout.data_type == data_types::i8 || input_node_layout.data_type == data_types::u8 ||
        input_node_layout.data_type == data_types::i32) {
        if (std::find(activations_int8.begin(), activations_int8.end(), func) == activations_int8.end())
            CLDNN_ERROR_MESSAGE(desc->id, "Requested activation is not supported for integer type.");
    }

    if (impl_param.has_fused_primitives()) {
        input_node_layout.data_type = impl_param.get_output_element_type();
    }

    return input_node_layout;
}

std::string activation_inst::to_string(activation_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream primitive_description;

    json_composite activation_info;
    activation_info.add("activation_func", static_cast<int>(desc->activation_function));
    activation_info.add("additional_params.a", desc->additional_params.a);
    activation_info.add("additional_params.b", desc->additional_params.b);
    activation_info.add("additional_params input", desc->additional_params_input);

    node_info->add("activation info", activation_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

activation_inst::typed_primitive_inst(network& network, activation_node const& node) : parent(network, node) {
    auto input_layout = node.get_input_layout();
    auto output_layout = node.get_output_layout();

    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "ReLU input rank",
                          input_layout.get_rank(),
                          "ReLU output rank",
                          output_layout.get_rank(),
                          "Relu input/output rank mismatch");
}
}  // namespace cldnn
