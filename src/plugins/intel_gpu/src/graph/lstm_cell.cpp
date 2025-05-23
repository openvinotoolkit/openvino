// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "lstm_cell_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(lstm_cell)

layout lstm_cell_inst::calc_output_layout(lstm_cell_node const& node, kernel_impl_params const& impl_param) {
    const auto& input_layout = impl_param.get_input_layout(0);
    const auto& input_pshape = input_layout.get_partial_shape();
    const auto& input_layout_hidden = impl_param.get_input_layout(1);
    const auto& input_pshape_hidden = input_layout_hidden.get_partial_shape();
    const auto& lstm_batch_size = input_pshape[0];
    const auto& lstm_hidden_size = input_pshape_hidden[1];

    return cldnn::layout{ov::PartialShape{lstm_batch_size, lstm_hidden_size}, input_layout.data_type, input_layout.format};
}

template<typename ShapeType>
std::vector<layout> lstm_cell_inst::calc_output_layouts(lstm_cell_node const& node, kernel_impl_params const& impl_param) {
    const auto& input_layout = impl_param.get_input_layout(0);
    const auto& input_pshape = input_layout.get_partial_shape();
    const auto& input_layout_hidden = impl_param.get_input_layout(1);
    const auto& input_pshape_hidden = input_layout_hidden.get_partial_shape();
    const auto& lstm_batch_size = input_pshape[0];
    const auto& lstm_hidden_size = input_pshape_hidden[1];

    auto out_layout = cldnn::layout{ShapeType{lstm_batch_size, lstm_hidden_size}, input_layout.data_type, input_layout.format};
    return {out_layout, out_layout};
}

template std::vector<layout> lstm_cell_inst::calc_output_layouts<ov::PartialShape>(lstm_cell_node const& node, const kernel_impl_params& impl_param);

std::string lstm_cell_inst::to_string(lstm_cell_node const& node) {
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite lstm_cell_info;
    node_info->add("lstm cell info", lstm_cell_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

lstm_cell_inst::typed_primitive_inst(network& network, lstm_cell_node const& node) : parent(network, node) {}
}  // namespace cldnn
