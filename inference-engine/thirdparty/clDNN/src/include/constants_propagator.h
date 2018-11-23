/*
// Copyright (c) 2017 Intel Corporation
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

#include "program_impl.h"
#include "data_inst.h"

namespace cldnn
{

class constants_propagator
{
public:
    constants_propagator(program_impl::ptr program);

    void visit_node(program_node& node);

    std::list<std::pair<primitive_id, memory_impl::ptr>> calculate();

private:
    program_impl::ptr prog;
    topology_impl tpl;
    std::list<typed_program_node<data>*> const_inputs;
    std::vector<primitive_id> const_outputs;
    bool has_non_trivial_constants = false;

    void handle_constant(program_node& node);
    void add_constant(program_node& node);
    void add_deps_to_tpl(const std::vector<program_node*>& node);
    bool is_already_in_tpl(const primitive_id& id);
};

}
