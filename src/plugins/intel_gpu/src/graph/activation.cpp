// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "activation_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>
#include <vector>

namespace cldnn {
primitive_type_id activation::type_id() {
    static primitive_type_base<activation> instance;
    return &instance;
}

layout activation_inst::calc_output_layout(activation_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for activation_node!");

    auto input_node_layout = node.input().get_non_padded_output_layout();
    auto func = node.get_primitive()->activation_function;

    std::vector<activation_func> activations_int8 = {
        activation_func::none,
        activation_func::negative,
        activation_func::negation,
        activation_func::relu,
        activation_func::floor,
        activation_func::clamp };

    if (input_node_layout.data_type == data_types::i8 || input_node_layout.data_type == data_types::i32) {
        if (std::find(activations_int8.begin(), activations_int8.end(), func) == activations_int8.end())
            CLDNN_ERROR_MESSAGE(node.id(), "Requested activation is not supported for integer type.");
    }

    if (node.has_fused_primitives()) {
        input_node_layout.data_type = node.get_fused_output_layout().data_type;
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
    auto input_layout = node.input().get_output_layout();
    auto output_layout = node.get_output_layout();

    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "ReLU input rank",
                          input_layout.get_rank(),
                          "ReLU output rank",
                          output_layout.get_rank(),
                          "Relu input/output rank mismatch");

    if (is_parameterized()) {
        /// Slope input x dimension should be equal to input feature size (one slope per channel).
        auto slope_layout = node.slope_input().get_output_layout();
        auto slope_input_size = slope_layout.size;
        auto input_feature_size = slope_layout.feature();

        CLDNN_ERROR_LESS_THAN(node.id(),
                              "Slope x size",
                              slope_input_size.feature[0],
                              "input feature size",
                              input_feature_size,
                              "Dimensions mismatch between input and slope input in Activation layer(slope x size "
                              "should be equal to input feature size)!");

        // All other dimensions should be 1
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "Slope input size count",
                              slope_input_size.count(),
                              "Slope input size x",
                              slope_input_size.feature[0],
                              "Dimensions mismatch of slope input in Activation layer!");
    }
}
}  // namespace cldnn
