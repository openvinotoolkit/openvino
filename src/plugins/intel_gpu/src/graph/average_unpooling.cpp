// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "average_unpooling_inst.h"
#include "primitive_type_base.h"
#include "sliding_window_utils_legacy.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id average_unpooling::type_id() {
    static primitive_type_base<average_unpooling> instance;
    return &instance;
}

layout average_unpooling_inst::calc_output_layout(average_unpooling_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for "
           "average_unpooling_node!");
    auto desc = node.get_primitive();

    auto input_layout = node.input().get_output_layout();

    auto stride = desc->stride;
    auto window_size = desc->size;

    CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(),
                                   "stride spatial X",
                                   stride.spatial[0],
                                   "",
                                   0,
                                   "Stride spatial X must be positive (>= 1)");
    CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(),
                                   "stride spatial Y",
                                   stride.spatial[1],
                                   "",
                                   0,
                                   "Stride spatial Y must be positive (>= 1)");
    CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(),
                                   "window size spatial X",
                                   window_size.spatial[0],
                                   "",
                                   0,
                                   "Size X (of pooling window) must be positive (>= 1)");
    CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(),
                                   "window size spatial Y",
                                   window_size.spatial[1],
                                   "",
                                   0,
                                   "Size Y (of pooling window) must be positive (>= 1)");

    tensor output_size(input_layout.size.batch[0],
                       input_layout.size.feature[0],
                       desc->output_size.spatial[0],
                       desc->output_size.spatial[1]);
    return {input_layout.data_type, input_layout.format, output_size};
}

std::string average_unpooling_inst::to_string(average_unpooling_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();
    auto& strd = desc->stride;
    auto& window_size = desc->size;

    std::stringstream primitive_description;

    json_composite average_unpooling_info;
    average_unpooling_info.add("input", input.id());
    average_unpooling_info.add("stride", strd.to_string());
    average_unpooling_info.add("window size", window_size.to_string());

    node_info->add("average_unpooling info", average_unpooling_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

average_unpooling_inst::typed_primitive_inst(network& network, average_unpooling_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
