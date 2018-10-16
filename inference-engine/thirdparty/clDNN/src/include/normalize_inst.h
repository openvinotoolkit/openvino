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
#include "api/CPP/normalize.hpp"
#include "primitive_inst.h"

namespace cldnn
{

template <>
struct typed_program_node<normalize> : public typed_program_node_base<normalize>
{
    using parent = typed_program_node_base<normalize>;

public:
    using parent::parent;

    decltype(auto) input() const { return get_dependency(0); }
    decltype(auto) scale() const { return get_dependency(1); }
};

using normalize_node = typed_program_node<normalize>;

template <>
class typed_primitive_inst<normalize> : public typed_primitive_inst_base<normalize>
{
    using parent = typed_primitive_inst_base<normalize>;

public:
    static layout calc_output_layout(normalize_node const& node);
    static std::string to_string(normalize_node const& node);
public:
    typed_primitive_inst(network_impl& network, normalize_node const& node);

    decltype(auto) scale_memory() const { return dep_memory(1); }
};

using normalize_inst = typed_primitive_inst<normalize>;

}
