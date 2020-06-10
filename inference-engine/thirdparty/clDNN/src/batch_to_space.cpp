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

#include "batch_to_space_inst.h"

#include "primitive_type_base.h"
#include "error_handler.h"
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

    const size_t dims_num = format::dimension(input_format);

    // Getting data from constant inputs. There are 3 args: block_shape, crops_begin, crops_end
    std::vector<std::vector<int32_t>> args;
    for (size_t i = 1; i < node.get_dependencies().size(); ++i) {
        auto& input = node.get_dependency(i).as<data>();
        auto& mem = input.get_attached_memory();
        std::vector<int32_t> sizes;
        if (input.get_output_layout().data_type == cldnn::data_types::i64) {
            int64_t* data = static_cast<int64_t*>(mem.lock());
            std::vector<int64_t> sizes_i64 = std::vector<int64_t>(data, data + input.get_output_layout().count());
            sizes.resize(sizes_i64.size());
            for (size_t j = 0; j < sizes.size(); j++)
                sizes[j] = static_cast<int32_t>(sizes_i64[j]);
        } else {
            int32_t* data = static_cast<int32_t*>(mem.lock());
            sizes = std::vector<int32_t>(data, data + input.get_output_layout().count());
        }
        args.push_back(sizes);
        mem.unlock();
    }

    const auto& block_shape = args[0];
    const auto& crops_begin = args[1];
    const auto& crops_end = args[2];

    if (block_shape[0] != 1)
        CLDNN_ERROR_MESSAGE(node.id(),
            "block_shape[0] is expected to be 1. Actual block_shape[0] is " +
            std::to_string(block_shape[0]));

    if (crops_begin[0] != 0)
        CLDNN_ERROR_MESSAGE(node.id(),
            "crops_begin[0] is expected to be 0. Actual crops_begin[0] is " +
            std::to_string(crops_begin[0]));

    if (crops_end[0] != 0)
        CLDNN_ERROR_MESSAGE(node.id(),
            "crops_end[0] is expected to be 0. Actual crops_end[0] is " +
            std::to_string(crops_end[0]));

    size_t block_sizes_multiplied = 1;
    for (const auto& el: block_shape)
        block_sizes_multiplied *= el;

    if (input_layout.size.batch[0] % block_sizes_multiplied != 0)
        CLDNN_ERROR_MESSAGE(node.id(),
            "The batch of the input tensor must be divisible by multiplied block sizes = " +
            std::to_string(block_sizes_multiplied));

    for(size_t i = 1; i < dims_num; ++i)
        if (crops_begin[i] + crops_end[i] >= block_shape[i] * input_layout.size.sizes(input_format)[i])
            CLDNN_ERROR_MESSAGE(node.id(),
                "Output dimensions must be positive");

    const auto output_batch = input_layout.size.batch[0] / block_sizes_multiplied;

    std::vector<int32_t> output_shape;
    output_shape.reserve(dims_num);
    for (size_t i = 1; i < dims_num; ++i) {
        output_shape.emplace_back(input_layout.size.sizes(input_format)[i] * block_shape[i] - crops_begin[i] - crops_end[i]);
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

std::string batch_to_space_inst::to_string(batch_to_space_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite batch_to_space_info;
    batch_to_space_info.add("input id", input.id());
    batch_to_space_info.add("block_shape id", node.get_dependency(1).id());
    batch_to_space_info.add("crops_begin id", node.get_dependency(2).id());
    batch_to_space_info.add("crops_end id", node.get_dependency(3).id());
    batch_to_space_info.add("block_shape shape", node.get_dependency(1).get_output_layout().size.to_string());
    batch_to_space_info.add("crops_begin shape", node.get_dependency(2).get_output_layout().size.to_string());
    batch_to_space_info.add("crops_end shape", node.get_dependency(3).get_output_layout().size.to_string());

    node_info->add("batch_to_space_info", batch_to_space_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

batch_to_space_inst::typed_primitive_inst(network_impl& network, batch_to_space_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
