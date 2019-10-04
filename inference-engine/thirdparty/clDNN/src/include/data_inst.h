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
#include "api/data.hpp"
#include "primitive_inst.h"
#include <string>
#include <memory>

namespace cldnn {

template <>
struct typed_program_node<data> : public typed_program_node_base<data> {
    using parent = typed_program_node_base<data>;

    typed_program_node(const std::shared_ptr<data> prim, program_impl& prog);

    memory_impl& get_attached_memory() const { return *mem; }
    void attach_memory(memory_impl& new_mem, bool invalidate_users_if_changed = true);

private:
    memory_impl::ptr mem;
};

using data_node = typed_program_node<data>;

template <>
class typed_primitive_inst<data> : public typed_primitive_inst_base<data> {
    using parent = typed_primitive_inst_base<data>;

public:
    static layout calc_output_layout(data_node const& node) { return node.get_attached_memory().get_layout(); }
    static std::string to_string(data_node const& node);

public:
    typed_primitive_inst(network_impl& network, data_node const& node);
};

using data_inst = typed_primitive_inst<data>;

}  // namespace cldnn
