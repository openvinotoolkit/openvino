// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "lstm_dynamic_input_inst.h"
#include "lstm_dynamic_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id lstm_dynamic_input::type_id() {
    static primitive_type_base<lstm_dynamic_input> instance;
    return &instance;
}
// input_tensor:   [b: batch, f: max_sequence_length, x: input_size, y: direction]
// weights_tensor: [b: 1, f: direction, x: input_size, y: 4 * hidden_size]
// output_tensor:  [b: batch, f: max_sequence_length, x: 4 * hidden_size, y: direction]
layout lstm_dynamic_input_inst::calc_output_layout(lstm_dynamic_input_node const& node) {
    assert(!node.get_primitive()->output_data_types.empty() &&
           "Output data type forcing is not supported for lstm_dynamic_node!");
    auto input_layout = node.input().get_output_layout();
    auto weight_layout = node.weights().get_output_layout();
    auto batch = input_layout.size.batch[0];
    auto direction = node.direction();
    auto output_sequence = input_layout.size.feature[0];
    return layout(input_layout.data_type,
                  input_layout.format,
                  tensor(batch, output_sequence, weight_layout.size.spatial[1], direction));
}

std::string lstm_dynamic_input_inst::to_string(lstm_dynamic_input_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto bias_id = desc->bias != "" ? desc->bias : "no bias";

    std::stringstream primitive_description;
    json_composite lstm_dynamic_input_info;
    lstm_dynamic_input_info.add("dyn_length id", desc->dyn_length);
    lstm_dynamic_input_info.add("weights id", desc->weights);
    lstm_dynamic_input_info.add("bias id", bias_id);
    lstm_dynamic_input_info.add("max seq len", node.input().get_output_layout().size.feature[0]);
    lstm_dynamic_input_info.add("hidden size", node.weights().get_output_layout().size.spatial[1] / 4);
    lstm_dynamic_input_info.add("direction", node.weights().get_output_layout().size.feature[0]);
    node_info->add("lstm_dynamic_input info", lstm_dynamic_input_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

lstm_dynamic_input_inst::typed_primitive_inst(network& network, lstm_dynamic_input_node const& node)
    : parent(network, node) {
    // Check input
    auto input_layout = node.input().get_output_layout();
    auto direction = node.direction();
    CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(),
                                  "input format",
                                  input_layout.format.value,
                                  "expected format",
                                  format::bfyx);
    lstm_dynamic_inst::check_direction(node.input(), direction, "input");

    // check dynamic length
    CLDNN_ERROR_BOOL(node.id(),
                     "Dynamic length memory",
                     !node.dyn_length_term(),
                     "Id of dynamic length memory is not set.");
    auto dyn_length_size = node.dyn_length().get_output_layout().count();
    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Batch",
                          node.get_output_layout().size.batch[0],
                          "Dynamic tensor elements count.",
                          dyn_length_size,
                          "Should be equal.");

    // check weights
    CLDNN_ERROR_BOOL(node.id(), "Weights memory", !node.weights_term(), "Id of weights memory is not set.");
    auto weights_id = node.weights().id();
    auto weights_tensor = node.weights().get_output_layout().size;
    auto hidden_size = weights_tensor.spatial[1] / 4;
    CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(),
                                  "weights format",
                                  node.weights().get_output_layout().format.value,
                                  "expected bfyx format",
                                  format::oiyx, format::lstm_weights_dio, format::bfyx);
    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Weights batch size",
                          weights_tensor.batch[0],
                          "1",
                          1,
                          "Sizes mismatch, weights_id: " + weights_id);
    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Weights x size",
                          weights_tensor.spatial[0],
                          "input_size",
                          input_layout.size.spatial[0],
                          "Sizes mismatch, weights_id: " + weights_id);

    // check bias
    if (node.bias_term()) {
        auto bias_id = node.id();
        auto bias_tensor = node.bias().get_output_layout().size;
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "Bias count",
                              bias_tensor.count(),
                              "direction * 4 * hidden_size",
                              direction * 4 * hidden_size,
                              "Bias count mismtach, bias_id: " + bias_id);
        lstm_dynamic_inst::check_direction(node.bias(), direction, "bias");
    }
}
}  // namespace cldnn
