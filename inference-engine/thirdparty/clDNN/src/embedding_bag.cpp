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

#include "embedding_bag_inst.h"

#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id embedding_bag::type_id() {
    static primitive_type_base<embedding_bag> instance;
    return &instance;
}

layout embedding_bag_inst::calc_output_layout(embedding_bag_node const& node) {
    auto desc = node.get_primitive();

    auto input_layout = node.input(0).get_output_layout();
    auto output_format = input_layout.format;

    auto output_shape = desc->output_shape;

    return layout(input_layout.data_type, output_format, output_shape);
}

std::string embedding_bag_inst::to_string(embedding_bag_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite embedding_bag_info;
    embedding_bag_info.add("input id", input.id());
    switch(desc->type) {
    case embedding_bag::packed_sum:
        embedding_bag_info.add("embedding bag type", "PackedSum");
        break;
    case embedding_bag::offsets_sum:
        embedding_bag_info.add("embedding bag type", "OffsetsSum");
        break;
    case embedding_bag::segments_sum:
        embedding_bag_info.add("embedding bag type", "SegmentsSum");
        break;
    }

    node_info->add("embedding_bag info", embedding_bag_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

embedding_bag_inst::typed_primitive_inst(network_impl& network, embedding_bag_node const& node)
    : parent(network, node) {}
}  // namespace cldnn
