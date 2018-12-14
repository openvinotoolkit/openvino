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

#include "softmax_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"

namespace cldnn
{
primitive_type_id softmax_type_id()
{
    static primitive_type_base<softmax> instance;
    return &instance;
}

layout softmax_inst::calc_output_layout(softmax_node const& node)
{
    return node.input().get_output_layout();
}

std::string softmax_inst::to_string(softmax_node const& node)
{
    auto desc      = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    node_info->dump(primitive_description);

    return primitive_description.str();
}

softmax_inst::typed_primitive_inst(network_impl& network, softmax_node const& node)
    : parent(network, node)
{
    //    auto& input_offset  = arg.input_offset;
    //    auto& output_offset = arg.output_offset;
    //    auto& output_size   = arg.output_size;
    //
    //    auto& input_inst  = arg.input[0].primitive().as<const memory&>().argument;
    //    auto& output_inst = arg.output[0].as<const memory&>().argument;
    //    for (auto &x : input_offset.raw) if (x < 0) throw std::runtime_error("Softmax negative input offset.");
    //
    //    for(size_t i = 0; i < input_inst.size.raw.size(); ++i) {
    //        if( input_inst.size.raw[i] < output_size.raw[i] +  input_offset.raw[i]) throw std::runtime_error("Softmax input/output size does not match.");
    //        if(output_inst.size.raw[i] < output_size.raw[i] + output_offset.raw[i]) throw std::runtime_error("Softmax sizes too small.");
    //    }

    //auto& input_inst = network.get_topology()->get_primitives().at(desc->input()[0]);
    //if (input_inst->output_layout->size.format == cldnn::format::bfyx)
    //    if (input_inst->output_layout->size.spatial[0] != 1 || input_inst->output_layout->size.spatial[1] != 1)
    //        throw std::runtime_error("Softmax input has more than one dimension per batch");
}
}
