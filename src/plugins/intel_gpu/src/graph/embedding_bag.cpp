// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "embedding_bag_inst.h"

#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
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
    switch (desc->type) {
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

embedding_bag_inst::typed_primitive_inst(network& network, embedding_bag_node const& node)
    : parent(network, node) {}
}  // namespace cldnn
