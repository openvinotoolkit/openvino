// Copyright (C) 2018-2024 Intel Corporation
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
