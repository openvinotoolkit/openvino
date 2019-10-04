/*
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
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "api/quantize.hpp"
#include "primitive_inst.h"
#include <string>

namespace cldnn {

template <>
struct typed_program_node<quantize> : public typed_program_node_base<quantize> {
    using parent = typed_program_node_base<quantize>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    size_t inputs_count() const { return get_dependencies().size(); }
    void set_output_data_type(data_types dt) { out_dt = dt; dt_changed = true; }
    data_types get_output_data_type() const { return out_dt; }
    bool has_custom_out_dt() const { return dt_changed; }

private:
    data_types out_dt;
    bool dt_changed = false;
};

using quantize_node = typed_program_node<quantize>;

template <>
class typed_primitive_inst<quantize> : public typed_primitive_inst_base<quantize> {
    using parent = typed_primitive_inst_base<quantize>;

public:
    static layout calc_output_layout(quantize_node const& node);
    static std::string to_string(quantize_node const& node);

public:
    typed_primitive_inst(network_impl& network, quantize_node const& desc);
};

using quantize_inst = typed_primitive_inst<quantize>;

}  // namespace cldnn
