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
#include "api/gemm.hpp"
#include "primitive_inst.h"
#include <string>

namespace cldnn {
template <>
struct typed_program_node<gemm> : public typed_program_node_base<gemm> {
    using parent = typed_program_node_base<gemm>;

public:
    using parent::parent;

    program_node& input(size_t idx = 0) const { return get_dependency(idx); }
    size_t inputs_count() const { return this->get_primitive()->input_size(); }
};

using gemm_node = typed_program_node<gemm>;

template <>
class typed_primitive_inst<gemm> : public typed_primitive_inst_base<gemm> {
    using parent = typed_primitive_inst_base<gemm>;

public:
    static layout calc_output_layout(gemm_node const& node);
    static std::string to_string(gemm_node const& node);

public:
    typed_primitive_inst(network_impl& network, gemm_node const& node);
};

using gemm_inst = typed_primitive_inst<gemm>;

}  // namespace cldnn
