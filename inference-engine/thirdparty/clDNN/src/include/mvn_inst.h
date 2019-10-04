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
#include "api/mvn.hpp"
#include "primitive_inst.h"
#include <string>

namespace cldnn {

template <>
struct typed_program_node<mvn> : public typed_program_node_base<mvn> {
    using parent = typed_program_node_base<mvn>;

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }
};

using mvn_node = typed_program_node<mvn>;

template <>
class typed_primitive_inst<mvn> : public typed_primitive_inst_base<mvn> {
    using parent = typed_primitive_inst_base<mvn>;

public:
    static layout calc_output_layout(mvn_node const& node);
    static std::string to_string(mvn_node const& node);

public:
    typed_primitive_inst(network_impl& network, mvn_node const& node);
};

using mvn_inst = typed_primitive_inst<mvn>;

}  // namespace cldnn
