// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "lstm_seq_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(lstm_seq)

layout lstm_seq_inst::calc_output_layout(lstm_seq_node const& node, kernel_impl_params const& impl_param) {
    assert(static_cast<bool>(impl_param.desc->output_data_types[0]) == false &&
           "Output data type forcing is not supported for lstm_seq_node!");
    auto input_layout = impl_param.get_input_layout();

    // tempGEMM{bfyx} = [b: batch, f: direction, x: 1,         y: 4 * hidden_size ] input
    // cell{bfyx}     = [b: batch, f: direction, x: 1,         y: hidden_size ] optional
    // output{bfyx}   = [b: batch, f: 2,         x: direction, y: hidden_size ] output
    // The output of the lstm_seq node is the concatenation of the intermediate [hidden, cell] tensors.
    // A crop/split node is needed to extract each individual tensors
    auto result =
        layout(input_layout.data_type,
               input_layout.format,
               tensor(input_layout.batch(), 2, input_layout.spatial(0) / 4, input_layout.feature()));
    return result;
}

template<typename ShapeType>
std::vector<layout> lstm_seq_inst::calc_output_layouts(lstm_seq_node const& node, kernel_impl_params const& impl_param) {
    std::vector<layout> output_layouts;

    // input partial shape [batch, input_size (= hidden_size * 4)]
    auto input_layout = impl_param.get_input_layout();
    auto input_pshape = input_layout.get_partial_shape();
    OPENVINO_ASSERT(static_cast<bool>(impl_param.desc->output_data_types[0]) == false, "Output data type forcing is not supported for lstm_seq_node!");
    OPENVINO_ASSERT(input_pshape.rank().get_length() == 2, "input_layout rank should be 2 on dynamic shape.");

    int lstm_input_size, lstm_batch_size, lstm_hidden_size;
    if (input_pshape[input_pshape.size() - 1].is_static()) {
        lstm_input_size = input_pshape[input_pshape.size() - 1].get_length();
        lstm_hidden_size = lstm_input_size / 4;
    } else {
        lstm_input_size = -1;
        lstm_hidden_size = -1;
    }

    if (input_pshape[input_pshape.size() - 2].is_static()) {
        lstm_batch_size = input_pshape[input_pshape.size() - 2].get_length();
    } else {
        lstm_batch_size = -1;
    }

    return {cldnn::layout{ov::PartialShape{lstm_batch_size, 2, 1, lstm_hidden_size}, input_layout.data_type, input_layout.format}};
}

template std::vector<layout> lstm_seq_inst::calc_output_layouts<ov::PartialShape>(lstm_seq_node const& node, const kernel_impl_params& impl_param);

std::string lstm_seq_inst::to_string(lstm_seq_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto cell_id = desc->cell;

    std::stringstream primitive_description;

    json_composite lstm_seq_info;
    lstm_seq_info.add("cell id", cell_id);
    node_info->add("lstm seq info", lstm_seq_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

lstm_seq_inst::typed_primitive_inst(network& network, lstm_seq_node const& node) : parent(network, node) {
    auto input_size = node.get_input_layout();
    CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(),
                                  "input format",
                                  input_size.format.value,
                                  "expected format",
                                  format::bfyx,
                                  format::fyxb);
}
}  // namespace cldnn
