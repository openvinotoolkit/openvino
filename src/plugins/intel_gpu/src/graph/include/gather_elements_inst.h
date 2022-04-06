/*
// Copyright (C) 2018-2022 Intel Corporation
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
#include "intel_gpu/primitives/gather_elements.hpp"
#include "primitive_inst.h"
#include <string>

namespace cldnn {
template <>
struct typed_program_node<gather_elements> : public typed_program_node_base<gather_elements> {
    using parent = typed_program_node_base<gather_elements>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return *get_dependency(index).first; }
};

using gather_elements_node = typed_program_node<gather_elements>;

template <>
class typed_primitive_inst<gather_elements> : public typed_primitive_inst_base<gather_elements> {
    using parent = typed_primitive_inst_base<gather_elements>;

public:
    static layout calc_output_layout(gather_elements_node const& node);
    static std::string to_string(gather_elements_node const& node);

public:
    typed_primitive_inst(network& network, gather_elements_node const& desc);
};

using gather_elements_inst = typed_primitive_inst<gather_elements>;
}  // namespace cldnn
