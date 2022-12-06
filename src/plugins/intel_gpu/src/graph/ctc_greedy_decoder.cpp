// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ctc_greedy_decoder_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(ctc_greedy_decoder)

layout ctc_greedy_decoder_inst::calc_output_layout(ctc_greedy_decoder_node const& node, kernel_impl_params const& impl_param) {
    auto input_node_layout = impl_param.get_input_layout();
    auto prim = impl_param.typed_desc<ctc_greedy_decoder>();
    auto output_type = prim->output_data_types[0].value_or(input_node_layout.data_type);

    return layout(output_type, input_node_layout.format, prim->output_tensor);
}

std::string ctc_greedy_decoder_inst::to_string(ctc_greedy_decoder_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();
    auto ctc_mr = desc->ctc_merge_repeated;
    auto blank_index = desc->blank_index;
    auto& input = node.input();
    auto& seq_ind = node.seq_indicators();

    std::stringstream primitive_description;

    json_composite ctc_gd_info;
    ctc_gd_info.add("input id", input.id());
    ctc_gd_info.add("seq inidicatior id", seq_ind.id());
    ctc_gd_info.add("ctc_mr", ctc_mr);
    ctc_gd_info.add("blank_index", blank_index);

    node_info->add("ctc_greedy_decoder info", ctc_gd_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

ctc_greedy_decoder_inst::typed_primitive_inst(network& network, ctc_greedy_decoder_node const& node) : parent(network, node) {}
}  // namespace cldnn
