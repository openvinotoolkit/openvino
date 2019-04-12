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

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "api/CPP/apply_adam.hpp"
#include "primitive_inst.h"

namespace cldnn
{

template <>
struct typed_program_node<apply_adam> : public typed_program_node_base<apply_adam>
{
    typed_program_node(const std::shared_ptr<apply_adam> prim, program_impl& prog);
    using parent = typed_program_node_base<apply_adam>;

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }
    program_node& m() const { return get_dependency(1); }
    program_node& v() const { return get_dependency(2); }
    program_node& beta1_power() const { return get_dependency(3); }
    program_node& beta2_power() const { return get_dependency(4); }
    program_node& additional_dep() const { return get_dependency(5); }

    bool has_additional_dep() const { return get_dependencies().size() > 5; }
};

using apply_adam_node = typed_program_node<apply_adam>;

template <>
class typed_primitive_inst<apply_adam> : public typed_primitive_inst_base<apply_adam>
{
    using parent = typed_primitive_inst_base<apply_adam>;

public:
    static layout calc_output_layout(apply_adam_node const& node);
    static std::string to_string(apply_adam_node const& node);

public:
    typed_primitive_inst(network_impl& network, apply_adam_node const& node);

    memory_impl& m_memory() const { return dep_memory(1); }
    memory_impl& v_memory() const { return dep_memory(2); }
    memory_impl& beta1_power_memory() const { return dep_memory(3); }
    memory_impl& beta2_power_memory() const { return dep_memory(4); }
    memory_impl& additional_dep() const { return dep_memory(5); }

    bool has_additional_dep() const { return _deps.size() > 5; }
};

using apply_adam_inst = typed_primitive_inst<apply_adam>;

}
