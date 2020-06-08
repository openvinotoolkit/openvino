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

#include "space_to_batch_inst.h"

#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"
#include "data_inst.h"
#include <string>
#include <vector>

namespace cldnn {
primitive_type_id cldnn::space_to_batch::type_id() {
    static primitive_type_base<space_to_batch> instance;
    return &instance;
}

layout space_to_batch_inst::calc_output_layout(space_to_batch_node const& node) {
    auto desc = node.get_primitive();

    auto input_layout = node.input(0).get_output_layout();
    auto input_format = input_layout.format;

    const size_t dims_num = format::dimension(input_format);

    // Getting data from constant inputs. There are 3 args: block_shape, pads_begin, pads_end
    std::vector<std::vector<int32_t>> args;
    for (size_t i = 1; i < node.get_dependencies().size(); ++i) {
        auto& input = node.get_dependency(i).as<data>();
        auto& mem = input.get_attached_memory();
        int32_t* data = static_cast<int32_t*>(mem.lock());
        std::vector<int32_t> sizes = std::vector<int32_t>(data, data + input.get_output_layout().count());
        args.push_back(sizes);
        mem.unlock();
    }

    const auto& block_shape = args[0];
    const auto& pads_begin = args[1];
    const auto& pads_end = args[2];

    if (block_shape[0] != 1)
        CLDNN_ERROR_MESSAGE(node.id(),
            "block_shape[0] is expected to be 1. Actual block_shape[0] is " +
            std::to_string(block_shape[0]));

    if (pads_begin[0] != 0)
        CLDNN_ERROR_MESSAGE(node.id(),
            "pads_begin[0] is expected to be 0. Actual pads_begin[0] is " +
            std::to_string(pads_begin[0]));

    if (pads_end[0] != 0)
        CLDNN_ERROR_MESSAGE(node.id(),
            "pads_end[0] is expected to be 0. Actual pads_end[0] is " +
            std::to_string(pads_end[0]));

    for (size_t i = 1; i < dims_num; ++i)
        if ((input_layout.size.sizes(input_format)[i] + pads_begin[i] + pads_end[i]) % block_shape[i] != 0)
            CLDNN_ERROR_MESSAGE(node.id(),
                "Input spatial shapes after padding must be divisible by block_shape");

    size_t block_sizes_multiplied = 1;
    for (const auto& el: block_shape)
        block_sizes_multiplied *= el;
    const auto output_batch = input_layout.size.batch[0] * block_sizes_multiplied;

    std::vector<int32_t> output_shape;
    output_shape.reserve(dims_num);
    for (size_t i = 1; i < dims_num; ++i) {
        output_shape.emplace_back((input_layout.size.sizes(input_format)[i] + pads_begin[i] + pads_end[i]) / block_shape[i]);
    }

    switch(input_format) {
        case format::bfzyx:
            return layout {
                input_layout.data_type,
                input_format,
                tensor(batch(output_batch), feature(output_shape[0]), spatial(output_shape[3], output_shape[2], output_shape[1]))};
        case format::bfwzyx:
            return layout {
                input_layout.data_type,
                input_format,
                tensor(batch(output_batch), feature(output_shape[0]), spatial(output_shape[4], output_shape[3],
                                                                              output_shape[2], output_shape[1]))};
        default:
            return layout {
                input_layout.data_type,
                input_format,
                tensor(batch(output_batch), feature(output_shape[0]), spatial(output_shape[2], output_shape[1]))};
    }
}

std::string space_to_batch_inst::to_string(space_to_batch_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite space_to_batch_info;
    space_to_batch_info.add("input id", input.id());
    space_to_batch_info.add("block_shape id", node.get_dependency(1).id());
    space_to_batch_info.add("pads_begin id", node.get_dependency(2).id());
    space_to_batch_info.add("pads_end id", node.get_dependency(3).id());
    space_to_batch_info.add("block_shape shape", node.get_dependency(1).get_output_layout().size.to_string());
    space_to_batch_info.add("pads_begin shape", node.get_dependency(2).get_output_layout().size.to_string());
    space_to_batch_info.add("pads_end shape", node.get_dependency(3).get_output_layout().size.to_string());

    node_info->add("space_to_batch_info", space_to_batch_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

space_to_batch_inst::typed_primitive_inst(network_impl& network, space_to_batch_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
