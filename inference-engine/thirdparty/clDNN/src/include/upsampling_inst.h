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
#include "api/CPP/upsampling.hpp"
#include "primitive_inst.h"
#include <memory>
#include "topology_impl.h"

namespace cldnn
{
template <>
struct typed_program_node<upsampling> : public typed_program_node_base<upsampling>
{
    using parent = typed_program_node_base<upsampling>;
    typed_program_node(const std::shared_ptr<upsampling> prim, program_impl& prog) : parent(prim, prog) { support_padding(true); }

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }
    program_node& input2() const { return get_dependency(1); }
};

using upsampling_node = typed_program_node<upsampling>;

template <>
class typed_primitive_inst<upsampling> : public typed_primitive_inst_base<upsampling>
{
    using parent = typed_primitive_inst_base<upsampling>;

public:
    static layout calc_output_layout(upsampling_node const& node);
    static std::string to_string(upsampling_node const& node);

public:
    typed_primitive_inst(network_impl& network, upsampling_node const& node);
};

using upsampling_inst = typed_primitive_inst<upsampling>;

}
