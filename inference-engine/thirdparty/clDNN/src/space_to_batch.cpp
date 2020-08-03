/*
// Copyright (c) 2020 Intel Corporation
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

#include "space_to_batch_inst.h"

#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"
#include "data_inst.h"
#include <string>
#include <vector>

namespace cldnn {
primitive_type_id cldnn::space_to_batch::type_id() {
    static primitive_type_base<space_to_batch> instance;
    return &instance;
}

layout space_to_batch_inst::calc_output_layout(space_to_batch_node const& node) {
    auto desc = node.get_primitive();

    auto input_layout = node.input(0).get_output_layout();
    auto input_format = input_layout.format;

    auto output_type = desc->output_data_type ? *desc->output_data_type : input_layout.data_type;

    if (node.has_fused_primitives())
        output_type = node.get_fused_output_layout().data_type;

    const size_t spatial_num = format::spatial_num(input_format);

    const auto& block_shape = desc->block_shape;
    const auto& pads_begin = desc->pads_begin;
    const auto& pads_end = desc->pads_end;

    if (block_shape.batch[0] != 1)
        CLDNN_ERROR_MESSAGE(node.id(),
            "block_shape[0] is expected to be 1. Actual block_shape[0] is " +
            std::to_string(block_shape.batch[0]));

    if (pads_begin.batch[0] != 0)
        CLDNN_ERROR_MESSAGE(node.id(),
            "pads_begin[0] is expected to be 0. Actual pads_begin[0] is " +
            std::to_string(pads_begin.batch[0]));

    if (pads_end.batch[0] != 0)
        CLDNN_ERROR_MESSAGE(node.id(),
            "pads_end[0] is expected to be 0. Actual pads_end[0] is " +
            std::to_string(pads_end.batch[0]));

    if ((input_layout.size.feature[0] + pads_begin.feature[0] + pads_end.feature[0]) % block_shape.feature[0] != 0)
            CLDNN_ERROR_MESSAGE(node.id(),
                "Input feature shape after padding must be divisible by block_shape");

    for (size_t i = 0; i < spatial_num; ++i)
        if ((input_layout.size.spatial[i] + pads_begin.spatial[i] + pads_end.spatial[i]) % block_shape.spatial[i] != 0)
            CLDNN_ERROR_MESSAGE(node.id(),
                "Input spatial shapes after padding must be divisible by block_shape");

    return layout{output_type, input_format, desc->out_size};
}

std::string space_to_batch_inst::to_string(space_to_batch_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite space_to_batch_info;
    space_to_batch_info.add("input id", input.id());

    node_info->add("space_to_batch_info", space_to_batch_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

space_to_batch_inst::typed_primitive_inst(network_impl& network, space_to_batch_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
