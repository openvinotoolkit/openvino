/*
// Copyright (c) 2019 Intel Corporation
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

#include "depth_to_space_inst.h"

#include "primitive_type_base.h"
#include "error_handler.h"
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

    if (input_layout.size.feature[0] % (block_size * block_size) != 0)
        CLDNN_ERROR_MESSAGE(
            node.id(),
            "The depth of the input tensor must be divisible by squared block size. Actual block size is " +
                std::to_string(block_size));

    const size_t feature = input_layout.size.feature[0] / block_size / block_size;
    const size_t y = input_layout.size.spatial[1] * block_size;
    const size_t x = input_layout.size.spatial[0] * block_size;

    return layout{
        input_layout.data_type,
        input_format,
        tensor(TensorValue(input_layout.size.batch[0]), TensorValue(feature), TensorValue(x), TensorValue(y))};
}

std::string depth_to_space_inst::to_string(depth_to_space_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite depth_to_space_info;
    depth_to_space_info.add("input id", input.id());
    depth_to_space_info.add("block size", desc->block_size);

    node_info->add("depth_to_space info", depth_to_space_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

depth_to_space_inst::typed_primitive_inst(network_impl& network, depth_to_space_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
