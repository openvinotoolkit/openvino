// Copyright (c) 2019 Intel Corporation
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

#pragma once

#include <api/gather_tree.hpp>

#include "primitive_inst.h"
#include <string>
#include <memory>

namespace cldnn {
template <>
struct typed_program_node<gather_tree> : typed_program_node_base<gather_tree> {
private:
    using parent = typed_program_node_base<gather_tree>;
public:
    using parent::parent;
    typed_program_node(const std::shared_ptr<gather_tree> prim, program_impl& prog) : parent(prim, prog) {
    }
    program_node& input() const { return get_dependency(0); }
};

using gather_tree_node = typed_program_node<gather_tree>;

template <>
class typed_primitive_inst<gather_tree> : public typed_primitive_inst_base<gather_tree> {
    using parent = typed_primitive_inst_base<gather_tree>;

public:
    static layout calc_output_layout(gather_tree_node const& node);
    static std::string to_string(gather_tree_node const& node);
    typed_primitive_inst(network_impl& network, gather_tree_node const& node);
};

using gather_tree_inst = typed_primitive_inst<gather_tree>;

}  // namespace cldnn
