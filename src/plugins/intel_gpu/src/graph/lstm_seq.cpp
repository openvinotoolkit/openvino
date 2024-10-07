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
    return lstm_seq_inst::calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template<typename ShapeType>
std::vector<layout> lstm_seq_inst::calc_output_layouts(lstm_seq_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<lstm_seq>();

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
    auto first_out_fmt = cldnn::format::bfyx;
    auto second_out_fmt = input_layout_x.format;
    auto third_out_fmt = input_layout_x.format;
    if (node.get_preferred_impl_type() == impl_types::onednn) {
        first_out_fmt = node.get_preferred_output_fmt();
        second_out_fmt = node.get_preferred_output_fmt(1);
        third_out_fmt = node.get_preferred_output_fmt(2);
        return {cldnn::layout{ShapeType{lstm_seq_length, lstm_batch_size, lstm_hidden_size, 1}, input_layout_x.data_type, first_out_fmt}, \
            cldnn::layout{ShapeType{lstm_batch_size, 1, lstm_hidden_size}, input_layout_x.data_type, second_out_fmt}, \
            cldnn::layout{ShapeType{lstm_batch_size, 1, lstm_hidden_size}, input_layout_x.data_type, third_out_fmt}};
    } else {
        return {cldnn::layout{ShapeTypelstm_batch_size, 1, {lstm_seq_length, lstm_hidden_size}, input_layout_x.data_type, first_out_fmt}, \
                cldnn::layout{ShapeType{lstm_batch_size, 1, lstm_hidden_size}, input_layout_x.data_type, second_out_fmt}, \
                cldnn::layout{ShapeType{lstm_batch_size, 1, lstm_hidden_size}, input_layout_x.data_type, third_out_fmt}};
    }
}

template std::vector<layout> lstm_seq_inst::calc_output_layouts<ov::PartialShape>(lstm_seq_node const& node, const kernel_impl_params& impl_param);

std::string lstm_seq_inst::to_string(lstm_seq_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite lstm_seq_info;
    node_info->add("lstm seq info", lstm_seq_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

lstm_seq_inst::typed_primitive_inst(network& network, lstm_seq_node const& node) : parent(network, node) {}
}  // namespace cldnn
