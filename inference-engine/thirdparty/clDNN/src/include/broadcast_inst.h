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

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <api/broadcast.hpp>

#include "primitive_inst.h"
#include <string>
#include <memory>

namespace cldnn {
template <>
struct typed_program_node<broadcast> : typed_program_node_base<broadcast> {
private:
    using parent = typed_program_node_base<broadcast>;

public:
    using parent::parent;

    typed_program_node(const std::shared_ptr<broadcast> prim, program_impl& prog) : parent(prim, prog) {
        support_padding_all(true);
    }
    program_node& input() const { return get_dependency(0); }
};

using broadcast_node = typed_program_node<broadcast>;

template <>
class typed_primitive_inst<broadcast> : public typed_primitive_inst_base<broadcast> {
    using parent = typed_primitive_inst_base<broadcast>;

public:
    static layout calc_output_layout(broadcast_node const& node);
    static std::string to_string(broadcast_node const& node);
    typed_primitive_inst(network_impl& network, broadcast_node const& node);
};

using broadcast_inst = typed_primitive_inst<broadcast>;
}  // namespace cldnn
