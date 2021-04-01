/*
// Copyright (c) 2020 Intel Corporation
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
#include "api/grn.hpp"
#include "primitive_inst.h"
#include <string>

namespace cldnn {

template <>
struct typed_program_node<grn> : public typed_program_node_base<grn> {
    using parent = typed_program_node_base<grn>;

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }
};

using grn_node = typed_program_node<grn>;

template <>
class typed_primitive_inst<grn> : public typed_primitive_inst_base<grn> {
    using parent = typed_primitive_inst_base<grn>;

public:
    static layout calc_output_layout(grn_node const& node);
    static std::string to_string(grn_node const& node);

public:
    typed_primitive_inst(network_impl& network, grn_node const& node);
};

using grn_inst = typed_primitive_inst<grn>;

}  // namespace cldnn
