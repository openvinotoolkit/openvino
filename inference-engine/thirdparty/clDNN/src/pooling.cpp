/*
// Copyright (c) 2016 Intel Corporation
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

#include "pooling_inst.h"
#include "primitive_type_base.h"
#include "sliding_window_utils.h"
#include "error_handler.h"
#include "json_object.h"

namespace cldnn
{
primitive_type_id pooling_type_id()
{
    static primitive_type_base<pooling> instance;
    return &instance;
}

layout pooling_inst::calc_output_layout(parent::typed_node const& node)
{
    auto desc = node.get_primitive();

    auto input_layout = node.input().get_output_layout();

    auto input_offset = desc->input_offset;
    auto stride = desc->stride;
    auto window_size = desc->size;

    if (!desc->argmax.empty())
        CLDNN_ERROR_NOT_EQUAL(node.id(), "Pooling mode", static_cast<size_t>(desc->mode), "should be max_with_argmax", static_cast<size_t>(pooling_mode::max_with_argmax), "Pooling mode should be set to max_with_argmax when argmax primitive is present.");

    if (desc->mode == pooling_mode::max_with_argmax)
    {
        CLDNN_ERROR_NOT_EQUAL(node.id(), "Argmax primitive", static_cast<size_t>(desc->argmax.empty()), "should not be empty", static_cast<size_t>(0), "Argmax primitive not present despite max_with_argmax mode.");
        
        auto argmax_layout = node.argmax().get_output_layout();
        CLDNN_ERROR_NOT_EQUAL(node.id(), "Argmax data type", static_cast<size_t>(argmax_layout.data_type), "expected to be fp32", static_cast<size_t>(data_types::f32), "Argmax data type is not fp32.");
        CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(), "Input_layout.format", input_layout.format.value, "argmax_layout.format", argmax_layout.format);
    }

    // TODO: Consider moving general parameter verification to arguments constructor.
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

    if (desc->with_output_size)
    {
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(), "User-defined size of output X", desc->output_size.spatial[0], "", 0, "User-defined size of output layout (spatial X) must be positive (>= 1)");
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(), "User-defined size of output Y", desc->output_size.spatial[1], "", 0, "User-defined size of output layout (spatial Y) must be positive (>= 1)");

        tensor output_size(input_layout.size.batch[0], input_layout.size.feature[0],
                           desc->output_size.spatial[0], desc->output_size.spatial[1]);
        return { input_layout.data_type, input_layout.format, output_size };
    }

    // TODO: Check compatibility of output size calculation (with caffe).
    auto output_range = calc_sliding_window_output_range<swor_mode::exceed_once_data>(
        input_layout.size, window_size, input_offset, stride, {1, 1, 1, 1}, true, 1);

    tensor output_size(input_layout.size.batch[0], input_layout.size.feature[0],
                       output_range.spatial[0], output_range.spatial[1]);
    return{ input_layout.data_type, input_layout.format, output_size };
}

std::string pooling_inst::to_string(pooling_node const& node)
{
    auto desc        = node.get_primitive();
    auto strd        = desc->stride;
    auto mode        = desc->mode == pooling_mode::max ? "max" : "average";
    auto node_info   = node.desc_to_json();
    auto kernel_size = desc->size;

    std::stringstream primitive_description;

    json_composite pooling_info;
    pooling_info.add("mode", mode);
    pooling_info.add("stride", strd.to_string());
    pooling_info.add("kernel size", kernel_size.to_string());
    pooling_info.add("input offset", desc->input_offset.to_string());
    if (desc->with_output_size)
    {
        json_composite ud_out_size_info;
        ud_out_size_info.add("size", desc->output_size.to_string());
        pooling_info.add("with_user_defined_output_size", ud_out_size_info);
    }

    node_info.add("pooling info", pooling_info);
    node_info.dump(primitive_description);

    return primitive_description.str();
}

}
