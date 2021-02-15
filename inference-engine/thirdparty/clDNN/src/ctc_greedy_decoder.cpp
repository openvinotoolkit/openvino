/*
// Copyright (c) 2020-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "ctc_greedy_decoder_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id ctc_greedy_decoder::type_id() {
    static primitive_type_base<ctc_greedy_decoder> instance;
    return &instance;
}

layout ctc_greedy_decoder_inst::calc_output_layout(ctc_greedy_decoder_node const& node) {
    auto input_node_layout = node.input().get_non_padded_output_layout();
    auto prim = node.get_primitive();
    auto output_type = prim->output_data_type ? *prim->output_data_type : input_node_layout.data_type;

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

ctc_greedy_decoder_inst::typed_primitive_inst(network_impl& network, ctc_greedy_decoder_node const& node) : parent(network, node) {}
}  // namespace cldnn
