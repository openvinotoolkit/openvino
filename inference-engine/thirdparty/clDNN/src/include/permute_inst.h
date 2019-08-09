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
#include "api/CPP/permute.hpp"
#include "primitive_inst.h"
#include <string>
#include <memory>

namespace cldnn {

template <>
struct typed_program_node<permute> : public typed_program_node_base<permute> {
    using parent = typed_program_node_base<permute>;
    typed_program_node(const std::shared_ptr<permute> prim, program_impl& prog) : parent(prim, prog) {
        support_padding_all(true);
    }

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }
};

using permute_node = typed_program_node<permute>;

template <>
class typed_primitive_inst<permute> : public typed_primitive_inst_base<permute> {
    using parent = typed_primitive_inst_base<permute>;

public:
    static layout calc_output_layout(permute_node const& node);
    static std::string to_string(permute_node const& node);

public:
    typed_primitive_inst(network_impl& network, permute_node const& node);
};

using permute_inst = typed_primitive_inst<permute>;

}  // namespace cldnn
