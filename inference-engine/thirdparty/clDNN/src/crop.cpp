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

#include "crop_inst.h"
#include "primitive_type_base.h"
#include "memory_impl.h"
#include "error_handler.h"
#include "json_object.h"

namespace cldnn
{
primitive_type_id crop_type_id()
{
    static primitive_type_base<crop> instance;
    return &instance;
}

layout crop_inst::calc_output_layout(crop_node const& node)
{
    auto input_layout = node.input().get_output_layout();
    auto result = layout({ input_layout.data_type, input_layout.format, node.get_primitive()->reference_input });
    return result;
}

std::string crop_inst::to_string(crop_node const& node)
{
    auto desc       = node.get_primitive();
    auto offsets    = desc->offsets;
    auto node_info  = node.desc_to_json();
    auto ref_input  = desc->reference_input;
    
    std::stringstream primitive_description;

    json_composite crop_info;
    crop_info.add("reference input", ref_input.to_string());
    crop_info.add("offset", offsets.to_string());    

    node_info->add("crop info", crop_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

crop_inst::typed_primitive_inst(network_impl& network, crop_node const& node)
    :parent(network, node)
{
    auto reference_input_sizes = argument.reference_input;
    auto inp_layout = node.input().get_output_layout();
    auto input_sizes = inp_layout.size;
    auto input_format = inp_layout.format;
    auto offsets = argument.offsets;

    CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(), "Input format", input_format.value, "supported crop input formats", format::yxfb, format::bfyx );

    //check if output sizes matches reference input sizes
    CLDNN_ERROR_TENSOR_SIZES_GREATER_THAN(node.id(), "Reference input", reference_input_sizes, "input sizes", input_sizes, "Reference input tensor/ input tensor mismtach");
    
    //check if offsets do not extend input sizes and if match the output sizes
    CLDNN_ERROR_TENSOR_SIZES_LESS_THAN(node.id(), "Batch offsets", offsets, "0 value", { 0, 0, 0, 0 }, "Invalid Batch offset: negative value");
    auto input_size_sub_offsets = input_sizes - offsets;
    CLDNN_ERROR_TENSOR_SIZES_LESS_THAN(node.id(), "input sizes - offsets", input_size_sub_offsets, "reference input sizes", reference_input_sizes, "Invalid Batch offset: exceeds data for output!");

    if (node.can_be_optimized())
    {
        build_deps();
        reuse_input();
    }
}


void crop_inst::on_execute()
{
    if (!node.can_be_optimized())
        return;

    if (_output && _network.get_engine().is_the_same_buffer(output_memory(), input_memory()))
        return;

    reuse_input();
}

void crop_inst::reuse_input()
{
    _output = _network.get_engine().reinterpret_buffer(input_memory(), node.get_output_layout());
}
}