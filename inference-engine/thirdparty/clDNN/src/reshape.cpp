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

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "reshape_inst.h"
#include "primitive_type_base.h"
#include "memory_impl.h"
#include "error_handler.h"
#include "json_object.h"

namespace cldnn
{

primitive_type_id reshape_type_id()
{
    static primitive_type_base<reshape> instance;
    return &instance;
}

layout reshape_inst::calc_output_layout(reshape_node const& node)
{
    auto input_layout = node.input().get_non_padded_output_layout();
    input_layout.size = node.get_primitive()->output_shape;
    return input_layout;
}

std::string reshape_inst::to_string(reshape_node const& node)
{
    auto desc      = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input    = node.input();

    std::stringstream primitive_description;

    json_composite reshape_info;
    reshape_info.add("input id", input.id());
    reshape_info.add("output shape", desc->output_shape);

    node_info->add("reshape info", reshape_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

reshape_inst::typed_primitive_inst(network_impl& network, reshape_node const& node)
    : parent(network, node, false)
{
    auto input_layout = node.input().get_output_layout();
    auto output_layout = node.get_output_layout();
    CLDNN_ERROR_DATA_TYPES_MISMATCH(node.id(), "Input layout data typr", input_layout.data_type, "output layout data type", output_layout.data_type, "");
    CLDNN_ERROR_NOT_EQUAL(node.id(), "Output layout count", output_layout.count(), "input layout count", input_layout.count(), "Output layout of reshape pirmitive changes size of input buffer");

    //if reshape operated in-place, postpone creation of the output until network run,
    //then create new memory object as the reinterpreted output of the previous primitive
    if (!node.is_in_place())
        _output = allocate_output();
    else
        reuse_input();
}

void reshape_inst::on_execute()
{
    if (!node.is_in_place())
        return;

    if (_output && _network.get_engine().is_the_same_buffer(output_memory(), input_memory()))
        return;

    reuse_input();
}

void reshape_inst::reuse_input()
{
    build_deps(); //reshape need deps
    _output = _network.get_engine().reinterpret_buffer(input_memory(), node.get_output_layout());
}

}