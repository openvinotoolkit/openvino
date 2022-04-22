// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "max_unpooling_inst.h"
#include "primitive_type_base.h"
#include "sliding_window_utils.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>
#include <memory>

namespace cldnn {
primitive_type_id max_unpooling::type_id() {
    static primitive_type_base<max_unpooling> instance;
    return &instance;
}

max_unpooling_node::typed_program_node(const std::shared_ptr<max_unpooling> prim, program& prog)
    : parent(prim, prog) {
    can_share_buffer(false);  // for max_unpooling initial zero values are significant
}

layout max_unpooling_inst::calc_output_layout(max_unpooling_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_types.at(0)) == false &&
           "Output data type forcing is not supported for max_unpooling_node!");
    auto desc = node.get_primitive();

    auto input_layout = node.input().get_output_layout();
    auto argmax_layout = node.argmax().get_output_layout();

    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Argmax data type",
                          static_cast<size_t>(argmax_layout.data_type),
                          "expected to be fp32",
                          static_cast<size_t>(data_types::f32),
                          "Argmax data type is not fp32.");

    if (desc->with_output_size) {
        tensor output_size(input_layout.size.batch[0],
                           input_layout.size.feature[0],
                           desc->output_size.spatial[0],
                           desc->output_size.spatial[1]);
        return {input_layout.data_type, input_layout.format, output_size};
    }

    auto pad = desc->pad;
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

    auto output_range = calc_sliding_window_needed_input_range(input_layout.size,
                                                               window_size,
                                                               pad,
                                                               stride,
                                                               {1, 1, 1, 1},
                                                               true,
                                                               1);

    tensor output_size(input_layout.size.batch[0],
                       input_layout.size.feature[0],
                       output_range.spatial[0],
                       output_range.spatial[1]);
    return {input_layout.data_type, input_layout.format, output_size};
}

std::string max_unpooling_inst::to_string(max_unpooling_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();
    auto& argmax = node.argmax();

    std::stringstream primitive_description;

    json_composite max_unmax_unpooling_info;
    max_unmax_unpooling_info.add("input", input.id());
    max_unmax_unpooling_info.add("argmax", argmax.id());

    node_info->add("max unmax_unpooling info", max_unmax_unpooling_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

max_unpooling_inst::typed_primitive_inst(network& network, max_unpooling_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
