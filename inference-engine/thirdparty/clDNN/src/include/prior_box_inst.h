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
#include "api/prior_box.hpp"
#include "primitive_inst.h"
#include <string>
#include <memory>

namespace cldnn {

template <>
struct typed_program_node<prior_box> : typed_program_node_base<prior_box> {
    using parent = typed_program_node_base<prior_box>;

    typed_program_node(std::shared_ptr<prior_box> prim, program_impl& prog);

    program_node& input() const { return get_dependency(0); }

    bool is_clustered() const { return get_primitive()->is_clustered(); }
    void calc_result();
    memory_impl::ptr get_result_buffer() const { return result; }

private:
    memory_impl::ptr result;
};

using prior_box_node = typed_program_node<prior_box>;

template <>
class typed_primitive_inst<prior_box> : public typed_primitive_inst_base<prior_box> {
    using parent = typed_primitive_inst_base<prior_box>;

public:
    static layout calc_output_layout(prior_box_node const& node);
    static std::string to_string(prior_box_node const& node);

public:
    typed_primitive_inst(network_impl& network, prior_box_node const& node);

    memory_impl& input_memory() const { return dep_memory(0); }
};

using prior_box_inst = typed_primitive_inst<prior_box>;

}  // namespace cldnn
