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

#include <api/contract.hpp>

#include "primitive_inst.h"
#include <string>
#include <memory>

namespace cldnn {
template <>
struct typed_program_node<contract> : typed_program_node_base<contract> {
private:
    using parent = typed_program_node_base<contract>;

public:
    using parent::parent;

    typed_program_node(const std::shared_ptr<contract> prim, program_impl& prog) : parent(prim, prog) {
        support_padding_all(true);
    }
    program_node& input() const { return get_dependency(0); }
};

using contract_node = typed_program_node<contract>;

template <>
class typed_primitive_inst<contract> : public typed_primitive_inst_base<contract> {
    using parent = typed_primitive_inst_base<contract>;

public:
    static layout calc_output_layout(contract_node const& node);
    static std::string to_string(contract_node const& node);
    typed_primitive_inst(network_impl& network, contract_node const& node);
};

using contract_inst = typed_primitive_inst<contract>;
}  // namespace cldnn
