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
#include "api/CPP/index_select.hpp"
#include "primitive_inst.h"

#include <memory>
#include <string>
#include <vector>

namespace cldnn {

template <>
struct typed_program_node<index_select> : public typed_program_node_base<index_select> {
    using parent = typed_program_node_base<index_select>;

public:
    typed_program_node(std::shared_ptr<primitive> prim, program_impl& prog) : parent(prim, prog) {}
    program_node& input() const { return get_dependency(0); }
    program_node& indices() const { return get_dependency(1); }
    bool get_reverse() const { return get_primitive()->reverse; }
    std::vector<index_select_axis_name> get_axes() const { return get_primitive()->axis; }
};

using index_select_node = typed_program_node<index_select>;

template <>
class typed_primitive_inst<index_select> : public typed_primitive_inst_base<index_select> {
    using parent = typed_primitive_inst_base<index_select>;

public:
    static layout calc_output_layout(index_select_node const& node);
    static std::string to_string(index_select_node const& node);
    typed_primitive_inst(network_impl& network, index_select_node const& node);

    memory_impl& input() const { return dep_memory(0); }
    memory_impl& indices() const { return dep_memory(1); }
    bool get_reverse() const { return node.get_reverse(); }
    std::vector<index_select_axis_name> get_axes() const { return node.get_axes(); }
};

using index_select_inst = typed_primitive_inst<index_select>;

}  // namespace cldnn
