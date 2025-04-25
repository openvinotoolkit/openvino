// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "gru_seq_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(gru_seq)

layout gru_seq_inst::calc_output_layout(gru_seq_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<gru_seq>();

    const auto& input_layout = impl_param.get_input_layout(0);
    const auto& input_pshape = input_layout.get_partial_shape();
    const auto& input_layout_hidden = impl_param.get_input_layout(1);
    const auto& input_pshape_hidden = input_layout_hidden.get_partial_shape();
    const auto& gru_batch_size = input_pshape[0];
    const auto& gru_seq_length = input_pshape[1];
    const auto& gru_hidden_size = input_pshape_hidden[2];

    auto first_out_fmt = cldnn::format::bfyx;
    if (node.get_preferred_output_fmt() != format::any) {
        first_out_fmt = node.get_preferred_output_fmt();
    }

    return cldnn::layout{ov::PartialShape{gru_batch_size, desc->num_directions(), gru_seq_length, gru_hidden_size}, input_layout.data_type, first_out_fmt};
}

template<typename ShapeType>
std::vector<layout> gru_seq_inst::calc_output_layouts(gru_seq_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<gru_seq>();

    const auto& input_layout = impl_param.get_input_layout(0);
    const auto& input_pshape = input_layout.get_partial_shape();
    const auto& input_layout_hidden = impl_param.get_input_layout(1);
    const auto& input_pshape_hidden = input_layout_hidden.get_partial_shape();
    const auto& gru_batch_size = input_pshape[0];
    const auto& gru_seq_length = input_pshape[1];
    const auto& gru_hidden_size = input_pshape_hidden[2];

    auto first_out_fmt = cldnn::format::bfyx;
    auto second_out_fmt = input_layout.format;
    if (node.get_preferred_output_fmt() != format::any) {
        first_out_fmt = node.get_preferred_output_fmt();
        second_out_fmt = node.get_preferred_output_fmt(1);
    }
    auto num_directions = desc->num_directions();

    return {cldnn::layout{ShapeType{gru_batch_size, num_directions, gru_seq_length, gru_hidden_size}, input_layout.data_type, first_out_fmt},
            cldnn::layout{ShapeType{gru_batch_size, num_directions, gru_hidden_size}, input_layout.data_type, second_out_fmt}};
}

template std::vector<layout> gru_seq_inst::calc_output_layouts<ov::PartialShape>(gru_seq_node const& node, const kernel_impl_params& impl_param);

std::string gru_seq_inst::to_string(gru_seq_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite gru_seq_info;
    node_info->add("gru seq info", gru_seq_info);
    node_info->add("linear before reset", desc->linear_before_reset);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

gru_seq_inst::typed_primitive_inst(network& network, gru_seq_node const& node) : parent(network, node) {}
}  // namespace cldnn
