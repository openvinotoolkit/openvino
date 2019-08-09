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
#include "api/CPP/split.hpp"
#include "primitive_inst.h"
#include <string>

namespace cldnn {

template <>
class typed_program_node<split> : public typed_program_node_base<split> {
    using parent = typed_program_node_base<split>;

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }
};

using split_node = typed_program_node<split>;

template <>
class typed_primitive_inst<split> : public typed_primitive_inst_base<split> {
    using parent = typed_primitive_inst_base<split>;

public:
    static layout calc_output_layout(split_node const& node);
    static std::string to_string(split_node const& node);
    typed_primitive_inst(network_impl& network, split_node const& node);
};

using split_inst = typed_primitive_inst<split>;
}  // namespace cldnn
