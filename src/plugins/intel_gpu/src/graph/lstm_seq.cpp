// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "lstm_seq_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>
#include "lstm_sequence_shape_inference.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(lstm_seq)

layout lstm_seq_inst::calc_output_layout(lstm_seq_node const& node, kernel_impl_params const& impl_param) {
    return lstm_seq_inst::calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template<typename ShapeType>
std::vector<layout> lstm_seq_inst::calc_output_layouts(lstm_seq_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<lstm_seq>();

    std::vector<ShapeType> input_shapes;
    for (size_t i = 0; i < desc->input.size(); ++i) {
        auto input_shape = impl_param.get_input_layout(i).get<ShapeType>();
        input_shapes.push_back(input_shape);
    }

    auto input_layout_x = impl_param.get_input_layout(0);
    auto input_pshape_x = input_layout_x.get_partial_shape();
    auto input_layout_hidden = impl_param.get_input_layout(1);
    auto input_pshape_hidden = input_layout_hidden.get_partial_shape();
    int lstm_batch_size, lstm_seq_length, lstm_hidden_size;
    if (input_pshape_x[0].is_static()) {
        lstm_batch_size = input_pshape_x[0].get_length();
    } else {
        lstm_batch_size = -1;
    }

    if (input_pshape_x[1].is_static()) {
        lstm_seq_length = input_pshape_x[1].get_length();
    } else {
        lstm_seq_length = -1;
    }

    if (input_pshape_hidden[2].is_static()) {
        lstm_hidden_size = input_pshape_hidden[2].get_length();
    } else {
        lstm_hidden_size = -1;
    }

    return {cldnn::layout{ShapeType{lstm_batch_size, 1, lstm_seq_length, lstm_hidden_size}, input_layout_x.data_type, input_layout_x.format}, \
            cldnn::layout{ShapeType{lstm_batch_size, 1, lstm_hidden_size}, input_layout_x.data_type, input_layout_x.format}, \
            cldnn::layout{ShapeType{lstm_batch_size, 1, lstm_hidden_size}, input_layout_x.data_type, input_layout_x.format}};
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
                                  format::bfyx);
}
}  // namespace cldnn
