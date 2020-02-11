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
#include "api/resample.hpp"
#include "primitive_inst.h"
#include <memory>
#include "topology_impl.h"
#include <string>

namespace cldnn {
template <>
struct typed_program_node<resample> : public typed_program_node_base<resample> {
    using parent = typed_program_node_base<resample>;
    typed_program_node(const std::shared_ptr<resample> prim, program_impl& prog) : parent(prim, prog) {
        support_padding_all(true);
    }

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }
    program_node& input2() const { return get_dependency(1); }
};

using resample_node = typed_program_node<resample>;

template <>
class typed_primitive_inst<resample> : public typed_primitive_inst_base<resample> {
    using parent = typed_primitive_inst_base<resample>;

public:
    static layout calc_output_layout(resample_node const& node);
    static std::string to_string(resample_node const& node);

public:
    typed_primitive_inst(network_impl& network, resample_node const& node);
};

using resample_inst = typed_primitive_inst<resample>;

}  // namespace cldnn
