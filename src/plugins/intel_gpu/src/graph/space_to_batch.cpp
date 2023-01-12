// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "space_to_batch_inst.h"

#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include "data_inst.h"
#include <string>
#include <vector>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(space_to_batch)

layout space_to_batch_inst::calc_output_layout(space_to_batch_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<space_to_batch>();

    auto input_layout = impl_param.get_input_layout();
    auto input_format = input_layout.format;

    auto output_type = desc->output_data_types[0].value_or(input_layout.data_type);

    if (impl_param.has_fused_primitives())
        output_type = impl_param.get_fused_output_layout().data_type;

    const size_t spatial_num = format::spatial_num(input_format);

    const auto& block_shape = desc->block_shape;
    const auto& pads_begin = desc->pads_begin;
    const auto& pads_end = desc->pads_end;

    if (block_shape.batch[0] != 1)
        CLDNN_ERROR_MESSAGE(desc->id,
            "block_shape[0] is expected to be 1. Actual block_shape[0] is " +
            std::to_string(block_shape.batch[0]));

    if (pads_begin.batch[0] != 0)
        CLDNN_ERROR_MESSAGE(desc->id,
            "pads_begin[0] is expected to be 0. Actual pads_begin[0] is " +
            std::to_string(pads_begin.batch[0]));

    if (pads_end.batch[0] != 0)
        CLDNN_ERROR_MESSAGE(desc->id,
            "pads_end[0] is expected to be 0. Actual pads_end[0] is " +
            std::to_string(pads_end.batch[0]));

    if ((input_layout.feature() + pads_begin.feature[0] + pads_end.feature[0]) % block_shape.feature[0] != 0)
            CLDNN_ERROR_MESSAGE(desc->id,
                "Input feature shape after padding must be divisible by block_shape");

    for (size_t i = 0; i < spatial_num; ++i)
        if ((input_layout.spatial(i) + pads_begin.spatial[i] + pads_end.spatial[i]) % block_shape.spatial[i] != 0)
            CLDNN_ERROR_MESSAGE(desc->id,
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

space_to_batch_inst::typed_primitive_inst(network& network, space_to_batch_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
