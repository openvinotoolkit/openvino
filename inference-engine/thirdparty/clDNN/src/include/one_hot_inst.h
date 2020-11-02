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

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <api/one_hot.hpp>

#include "primitive_inst.h"
#include <string>
#include <memory>

namespace cldnn {
template <>
struct typed_program_node<one_hot> : typed_program_node_base<one_hot> {
private:
    using parent = typed_program_node_base<one_hot>;

public:
    using parent::parent;

    typed_program_node(const std::shared_ptr<one_hot> prim, program_impl& prog) : parent(prim, prog) {
        support_padding_all(true);
    }
    program_node& input() const { return get_dependency(0); }
};

using one_hot_node = typed_program_node<one_hot>;

template <>
class typed_primitive_inst<one_hot> : public typed_primitive_inst_base<one_hot> {
    using parent = typed_primitive_inst_base<one_hot>;

public:
    static layout calc_output_layout(one_hot_node const& node);
    static std::string to_string(one_hot_node const& node);
    typed_primitive_inst(network_impl& network, one_hot_node const& node);
};

using one_hot_inst = typed_primitive_inst<one_hot>;
}  // namespace cldnn
