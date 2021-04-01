/*
// Copyright (c) 2018 Intel Corporation
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

#include "average_unpooling_inst.h"
#include "primitive_type_base.h"
#include "sliding_window_utils.h"
#include "error_handler.h"
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

average_unpooling_inst::typed_primitive_inst(network_impl& network, average_unpooling_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
