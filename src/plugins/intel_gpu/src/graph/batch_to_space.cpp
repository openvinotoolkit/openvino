// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "batch_to_space_inst.h"

#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include "data_inst.h"
#include <string>
#include <vector>

namespace cldnn {
primitive_type_id cldnn::batch_to_space::type_id() {
    static primitive_type_base<batch_to_space> instance;
    return &instance;
}

layout batch_to_space_inst::calc_output_layout(batch_to_space_node const& node) {
    auto desc = node.get_primitive();

    auto input_layout = node.input(0).get_output_layout();
    auto input_format = input_layout.format;

    auto output_type = !desc->output_data_types.empty() ? *desc->output_data_types.at(0) : input_layout.data_type;

    if (node.has_fused_primitives())
        output_type = node.get_fused_output_layout().data_type;

    const size_t spatial_num = format::spatial_num(input_format);

    const auto& block_shape = desc->block_shape;
    const auto& crops_begin = desc->crops_begin;
    const auto& crops_end = desc->crops_end;

    if (block_shape.batch[0] != 1)
        CLDNN_ERROR_MESSAGE(node.id(),
            "block_shape[0] is expected to be 1. Actual block_shape[0] is " +
            std::to_string(block_shape.batch[0]));

    if (crops_begin.batch[0] != 0)
        CLDNN_ERROR_MESSAGE(node.id(),
            "crops_begin[0] is expected to be 0. Actual crops_begin[0] is " +
            std::to_string(crops_begin.batch[0]));

    if (crops_end.batch[0] != 0)
        CLDNN_ERROR_MESSAGE(node.id(),
            "crops_end[0] is expected to be 0. Actual crops_end[0] is " +
            std::to_string(crops_end.batch[0]));

    size_t block_sizes_multiplied = block_shape.feature[0];
    for (size_t i = 0; i < spatial_num; ++i)
        block_sizes_multiplied *= block_shape.spatial[i];

    if (input_layout.size.batch[0] % block_sizes_multiplied != 0)
        CLDNN_ERROR_MESSAGE(node.id(),
            "The batch of the input tensor must be divisible by multiplied block sizes = " +
            std::to_string(block_sizes_multiplied));

    if (crops_begin.feature[0] + crops_end.feature[0] >= block_shape.feature[0] * input_layout.size.feature[0])
            CLDNN_ERROR_MESSAGE(node.id(),
                "Output dimensions must be positive");

    for (size_t i = 0; i < spatial_num; ++i)
        if (crops_begin.spatial[i] + crops_end.spatial[i] >= block_shape.spatial[i] * input_layout.size.spatial[i])
            CLDNN_ERROR_MESSAGE(node.id(),
                "Output dimensions must be positive");

    return layout{output_type, input_format, desc->out_size};
}

std::string batch_to_space_inst::to_string(batch_to_space_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite batch_to_space_info;
    batch_to_space_info.add("input id", input.id());

    node_info->add("batch_to_space_info", batch_to_space_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

batch_to_space_inst::typed_primitive_inst(network& network, batch_to_space_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
