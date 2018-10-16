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
#pragma once
#include "api/CPP/input_layout.hpp"
#include "primitive_inst.h"

namespace cldnn
{
struct memory_impl;

template <>
struct typed_program_node<input_layout> : public typed_program_node_base<input_layout>
{
    using parent = typed_program_node_base<input_layout>;
    using parent::parent;
};

using input_layout_node = typed_program_node<input_layout>;

template <>
class typed_primitive_inst<input_layout> : public typed_primitive_inst_base<input_layout>
{
    using parent = typed_primitive_inst_base<input_layout>;

public:
    static layout calc_output_layout(input_layout_node const& node)
    {
        return node.get_primitive()->layout;
    }
    static std::string to_string(input_layout_node const& node);

public:
    typed_primitive_inst(network_impl& network, input_layout_node const& node);

    void set_data(memory_impl& mem);
};

using input_layout_inst = typed_primitive_inst<input_layout>;

}
