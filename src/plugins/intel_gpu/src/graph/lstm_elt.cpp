// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "lstm_elt_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(lstm_elt)

layout lstm_elt_inst::calc_output_layout(lstm_elt_node const& node, kernel_impl_params const& impl_param) {
    assert(static_cast<bool>(impl_param.desc->output_data_types[0]) == false &&
           "Output data type forcing is not supported for lstm_elt_node!");
    auto input_layout = impl_param.get_input_layout();

    // tempGEMM{bfyx} = [b: batch, f: direction, x: 1,         y: 4 * hidden_size ] input
    // cell{bfyx}     = [b: batch, f: direction, x: 1,         y: hidden_size ] optional
    // output{bfyx}   = [b: batch, f: 2,         x: direction, y: hidden_size ] output
    // The output of the lstm_elt node is the concatenation of the intermediate [hidden, cell] tensors.
    // A crop/split node is needed to extract each individual tensors
    auto result =
        layout(input_layout.data_type,
               input_layout.format,
               tensor(input_layout.batch(), 2, input_layout.spatial(0) / 4, input_layout.feature()));
    return result;
}

template<typename ShapeType>
std::vector<layout> lstm_elt_inst::calc_output_layouts(lstm_elt_node const& node, kernel_impl_params const& impl_param) {
    std::vector<layout> output_layouts;

    // input partial shape [batch, input_size (= hidden_size * 4)]
    auto input_layout = impl_param.get_input_layout();
    auto input_pshape = input_layout.get_partial_shape();
    OPENVINO_ASSERT(static_cast<bool>(impl_param.desc->output_data_types[0]) == false, "Output data type forcing is not supported for lstm_elt_node!");
    OPENVINO_ASSERT(input_pshape.rank().get_length() == 2, "input_layout rank should be 2 on dynamic shape.");

    auto lstm_input_size = static_cast<cldnn::tensor::value_type>(input_pshape[1].get_length());
    auto lstm_batch_size = static_cast<cldnn::tensor::value_type>(input_pshape[0].get_length());
    auto lstm_hidden_size = static_cast<cldnn::tensor::value_type>(lstm_input_size / 4);

    return {cldnn::layout{ov::PartialShape{lstm_batch_size, 2, 1, lstm_hidden_size}, input_layout.data_type, input_layout.format}};
}

template std::vector<layout> lstm_elt_inst::calc_output_layouts<ov::PartialShape>(lstm_elt_node const& node, const kernel_impl_params& impl_param);

std::string lstm_elt_inst::to_string(lstm_elt_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto cell_id = desc->cell;

    std::stringstream primitive_description;

    json_composite lstm_elt_info;
    lstm_elt_info.add("cell id", cell_id);
    node_info->add("lstm elt info", lstm_elt_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

lstm_elt_inst::typed_primitive_inst(network& network, lstm_elt_node const& node) : parent(network, node) {
    auto input_size = node.input().get_output_layout();
    CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(),
                                  "input format",
                                  input_size.format.value,
                                  "expected format",
                                  format::bfyx,
                                  format::fyxb);
}
}  // namespace cldnn
