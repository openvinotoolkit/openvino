// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "batch_to_space_inst.h"

#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>
#include <vector>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(batch_to_space)

layout batch_to_space_inst::calc_output_layout(batch_to_space_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<batch_to_space>();

    auto input_layout = impl_param.get_input_layout();
    auto input_format = input_layout.format;

    auto output_type = desc->output_data_types[0].value_or(input_layout.data_type);

    if (impl_param.has_fused_primitives())
        output_type = impl_param.get_fused_output_layout().data_type;

    const size_t spatial_num = format::spatial_num(input_format);

    const auto& block_shape = desc->block_shape;
    const auto& crops_begin = desc->crops_begin;
    const auto& crops_end = desc->crops_end;

    if (block_shape.batch[0] != 1)
        CLDNN_ERROR_MESSAGE(desc->id,
            "block_shape[0] is expected to be 1. Actual block_shape[0] is " +
            std::to_string(block_shape.batch[0]));

    if (crops_begin.batch[0] != 0)
        CLDNN_ERROR_MESSAGE(desc->id,
            "crops_begin[0] is expected to be 0. Actual crops_begin[0] is " +
            std::to_string(crops_begin.batch[0]));

    if (crops_end.batch[0] != 0)
        CLDNN_ERROR_MESSAGE(desc->id,
            "crops_end[0] is expected to be 0. Actual crops_end[0] is " +
            std::to_string(crops_end.batch[0]));

    size_t block_sizes_multiplied = block_shape.feature[0];
    for (size_t i = 0; i < spatial_num; ++i)
        block_sizes_multiplied *= block_shape.spatial[i];

    if (input_layout.batch() % block_sizes_multiplied != 0)
        CLDNN_ERROR_MESSAGE(desc->id,
            "The batch of the input tensor must be divisible by multiplied block sizes = " +
            std::to_string(block_sizes_multiplied));

    if (crops_begin.feature[0] + crops_end.feature[0] >= block_shape.feature[0] * input_layout.feature())
            CLDNN_ERROR_MESSAGE(desc->id,
                "Output dimensions must be positive");

    for (size_t i = 0; i < spatial_num; ++i)
        if (crops_begin.spatial[i] + crops_end.spatial[i] >= block_shape.spatial[i] * input_layout.spatial(i))
            CLDNN_ERROR_MESSAGE(desc->id,
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
