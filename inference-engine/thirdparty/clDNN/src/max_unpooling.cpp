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

#include "max_unpooling_inst.h"
#include "primitive_type_base.h"
#include "sliding_window_utils.h"
#include "error_handler.h"
#include "json_object.h"

namespace cldnn
{
primitive_type_id max_unpooling_type_id()
{
    static primitive_type_base<max_unpooling> instance;
    return &instance;
}

layout max_unpooling_inst::calc_output_layout(max_unpooling_node const& node)
{
    auto desc = node.get_primitive();

    auto input_layout = node.input().get_output_layout();
    auto argmax_layout = node.argmax().get_output_layout();

    CLDNN_ERROR_NOT_EQUAL(node.id(), "Argmax data type", static_cast<size_t>(argmax_layout.data_type), "expected to be fp32", static_cast<size_t>(data_types::f32), "Argmax data type is not fp32.");

    if (desc->with_output_size)
    {
        tensor output_size(input_layout.size.batch[0], input_layout.size.feature[0],
            desc->output_size.spatial[0], desc->output_size.spatial[1]);
        return{ input_layout.data_type, input_layout.format, output_size };
    }

    auto input_offset = desc->input_offset;
    auto stride = desc->stride;
    auto window_size = desc->size;

    CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(), "stride spatial X", stride.spatial[0], "", 0, "Stride spatial X must be positive (>= 1)");
    CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(), "stride spatial Y", stride.spatial[1], "", 0, "Stride spatial Y must be positive (>= 1)");
    CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(), "window size spatial X", window_size.spatial[0], "", 0, "Size X (of pooling window) must be positive (>= 1)");
    CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(), "window size spatial Y", window_size.spatial[1], "", 0, "Size Y (of pooling window) must be positive (>= 1)");
    CLDNN_ERROR_GREATER_THAN(node.id(), "Input offset spatial X", 2 * input_offset.spatial[0], "input layout size spatial X", input_layout.size.spatial[0], "Input offset is greater than input data range. There is no input data to process");
    CLDNN_ERROR_GREATER_THAN(node.id(), "Input offset spatial Y", 2 * input_offset.spatial[1], "input layout size spatial Y", input_layout.size.spatial[1], "Input offset is greater than input data range. There is no input data to process");
    CLDNN_ERROR_GREATER_THAN(node.id(), "Negate input offset spatial X", -input_offset.spatial[0], "input window size spatial X", window_size.spatial[0], "First pool is outside of image. please reduce input offset X");
    CLDNN_ERROR_GREATER_THAN(node.id(), "Negate input offset spatial Y", -input_offset.spatial[1], "input window size spatial Y", window_size.spatial[1], "First pool is outside of image. please reduce input offset Y");
    CLDNN_ERROR_NOT_EQUAL(node.id(), "Input offset feature", input_offset.feature[0], "", 0, "Input offset in feature is not supported");
    CLDNN_ERROR_NOT_EQUAL(node.id(), "Input offset batch", input_offset.batch[0], "", 0, "Input offset in batch is not supported");

    auto output_range = calc_sliding_window_needed_input_range(
        input_layout.size, window_size, input_offset, stride, { 1, 1, 1, 1 }, true, 1);

    tensor output_size(input_layout.size.batch[0], input_layout.size.feature[0],
        output_range.spatial[0], output_range.spatial[1]);
    return{ input_layout.data_type, input_layout.format, output_size };
}

std::string max_unpooling_inst::to_string(max_unpooling_node const& node)
{
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();
    auto& argmax = node.argmax();

    std::stringstream primitive_description;

    json_composite max_unmax_unpooling_info;
    max_unmax_unpooling_info.add("input", input.id());
    max_unmax_unpooling_info.add("argmax", argmax.id());

    node_info.add("max unmax_unpooling info", max_unmax_unpooling_info);
    node_info.dump(primitive_description);

    return primitive_description.str();
}

max_unpooling_inst::typed_primitive_inst(network_impl& network, max_unpooling_node const& node)
    :parent(network, node)
{
}

}
