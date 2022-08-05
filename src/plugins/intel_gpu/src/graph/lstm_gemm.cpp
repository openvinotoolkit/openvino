// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "lstm_gemm_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id lstm_gemm::type_id() {
    static primitive_type_base<lstm_gemm> instance;
    return &instance;
}

layout lstm_gemm_inst::calc_output_layout(lstm_gemm_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for lstm_gemm_node!");
    auto desc = node.get_primitive();
    auto input_layout = node.input().get_output_layout();
    auto weights_layout = node.weights().get_output_layout();

    //   input{bfyx}     = [b: batch, f: sequence,   x: input_size,      y: 1]
    //   weights{bfyx}   = [b: 1,     f: direction,  x: 4 * hidden_size, y: input_size ]
    //   recurrent{bfyx} = [b: 1,     f: direction,  x: 4 * hidden_size, y: hidden_size ]
    //   biases{bfyx}    = [b: 1,     f:1 ,          x: direction,       y:  4 * hidden_size ]
    //   hidden{bfyx}    = [b: batch, f:  direction, x: 1 ,              y: hidden_size ] optional
    //   tempGEMM{bfyx}  = [b: batch, f: direction,  x: 4*hidden_size,   y: 1] output
    auto result =
        layout(input_layout.data_type,
               input_layout.format,
               tensor(input_layout.batch(), weights_layout.feature(), weights_layout.spatial(1), 1));
    return result;
}

std::string lstm_gemm_inst::to_string(lstm_gemm_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto weights_id = desc->weights;
    auto recurrent_id = desc->recurrent;
    auto bias_id = desc->bias != "" ? desc->bias : "no bias";
    auto hidden_id = desc->hidden != "" ? desc->hidden : "no inital hidden";

    std::stringstream primitive_description;

    json_composite lstm_gemm_info;
    lstm_gemm_info.add("weights id", weights_id);
    lstm_gemm_info.add("recurrent id", recurrent_id);
    lstm_gemm_info.add("bias id", bias_id);
    lstm_gemm_info.add("hidden id", hidden_id);
    node_info->add("lstm gemm info", lstm_gemm_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

lstm_gemm_inst::typed_primitive_inst(network& network, lstm_gemm_node const& node) : parent(network, node) {
    auto input_layout = node.input().get_output_layout();
    CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(),
                                  "input format",
                                  input_layout.format.value,
                                  "expected format",
                                  format::bfyx,
                                  format::fyxb);
}
}  // namespace cldnn
