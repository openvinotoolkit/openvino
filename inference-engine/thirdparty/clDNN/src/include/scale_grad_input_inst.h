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
#include "api/CPP/scale_grad_input.hpp"
#include "primitive_inst.h"
#include <string>

namespace cldnn {

template <>
struct typed_program_node<scale_grad_input> : public typed_program_node_base<scale_grad_input> {
    using parent = typed_program_node_base<scale_grad_input>;

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }
    program_node& scale_in() const { return get_dependency(1); }
};

using scale_grad_input_node = typed_program_node<scale_grad_input>;

template <>
class typed_primitive_inst<scale_grad_input> : public typed_primitive_inst_base<scale_grad_input> {
    using parent = typed_primitive_inst_base<scale_grad_input>;

public:
    static layout calc_output_layout(scale_grad_input_node const& node);
    static std::string to_string(scale_grad_input_node const& node);

public:
    typed_primitive_inst(network_impl& network, scale_grad_input_node const& desc);

    memory_impl& scale_input_memory() const { return dep_memory(1); }
};

using scale_grad_input_inst = typed_primitive_inst<scale_grad_input>;

}  // namespace cldnn
