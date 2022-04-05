// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "depth_to_space_inst.h"

#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id depth_to_space::type_id() {
    static primitive_type_base<depth_to_space> instance;
    return &instance;
}

layout depth_to_space_inst::calc_output_layout(depth_to_space_node const& node) {
    auto desc = node.get_primitive();

    auto input_layout = node.input(0).get_output_layout();
    auto input_format = input_layout.format;

    const size_t block_size = desc->block_size;

    if (block_size < 2)
        CLDNN_ERROR_MESSAGE(node.id(),
                            "Invalid depthToSpace block_size value (should equal at least two). Actual block size is" +
                                std::to_string(block_size));

    if (input_layout.feature() % (block_size * block_size) != 0)
        CLDNN_ERROR_MESSAGE(
            node.id(),
            "The depth of the input tensor must be divisible by squared block size. Actual block size is " +
                std::to_string(block_size));

    auto out_size = input_layout.size;
    if (format::spatial_num(input_layout.format) == 3) {
        const size_t feature = input_layout.feature() / block_size / block_size / block_size;
        const size_t z = input_layout.spatial(2) * block_size;
        const size_t y = input_layout.spatial(1) * block_size;
        const size_t x = input_layout.spatial(0) * block_size;
        out_size = tensor(TensorValue(input_layout.batch()), TensorValue(feature), TensorValue(x), TensorValue(y), TensorValue(z));
    } else {
        const size_t feature = input_layout.feature() / block_size / block_size;
        const size_t y = input_layout.spatial(1) * block_size;
        const size_t x = input_layout.spatial(0) * block_size;
        out_size = tensor(TensorValue(input_layout.batch()), TensorValue(feature), TensorValue(x), TensorValue(y));
    }

    if (node.has_fused_primitives()) {
        input_layout.data_type = node.get_fused_output_layout().data_type;
    }

    return layout{input_layout.data_type, input_format, out_size};
}

std::string depth_to_space_inst::to_string(depth_to_space_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite depth_to_space_info;
    depth_to_space_info.add("input id", input.id());
    depth_to_space_info.add("block size", desc->block_size);
    depth_to_space_info.add("mode", desc->mode == depth_to_space_mode::blocks_first ? "blocks_first" : "depth_first");

    node_info->add("depth_to_space info", depth_to_space_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

depth_to_space_inst::typed_primitive_inst(network& network, depth_to_space_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
