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
#include "api_extension/lstm_dynamic_input.hpp"
#include "primitive_inst.h"
#include "error_handler.h"
#include <memory>
#include <string>

namespace cldnn {

template <>
struct typed_program_node<lstm_dynamic_input> : public typed_program_node_base<lstm_dynamic_input> {
    using parent = typed_program_node_base<lstm_dynamic_input>;

public:
    typed_program_node(std::shared_ptr<primitive> prim, program_impl& prog) : parent(prim, prog) {}

    program_node& input() const { return get_dependency(0); }
    program_node& dyn_length() const { return get_dependency(1); }
    program_node& weights() const { return get_dependency(2); }

    program_node& bias() const {
        CLDNN_ERROR_BOOL(id(), "Bias term", !bias_term(), "Trying to get non existing bias.");
        return get_dependency(3);
    }

    int32_t direction() const { return weights().get_output_layout().size.feature[0]; }
    bool dyn_length_term() const { return !get_primitive()->dyn_length.empty(); }
    bool bias_term() const { return !get_primitive()->bias.empty(); }
    bool weights_term() const { return !get_primitive()->weights.empty(); }
};

using lstm_dynamic_input_node = typed_program_node<lstm_dynamic_input>;

template <>
class typed_primitive_inst<lstm_dynamic_input> : public typed_primitive_inst_base<lstm_dynamic_input> {
    using parent = typed_primitive_inst_base<lstm_dynamic_input>;

public:
    static layout calc_output_layout(lstm_dynamic_input_node const& node);
    static std::string to_string(lstm_dynamic_input_node const& node);

public:
    typed_primitive_inst(network_impl& network, lstm_dynamic_input_node const& node);

    memory_impl& dyn_length_memory() const { return dep_memory(1); }
    memory_impl& weights_memory() const { return dep_memory(2); }
    memory_impl& bias_memory() const {
        CLDNN_ERROR_BOOL(id(), "Bias term", !bias_term(), "Trying to get non existing bias memory.");
        return dep_memory(3);
    }
    int32_t direction() const { return node.direction(); }
    bool bias_term() const { return node.bias_term(); }
};

using lstm_dynamic_input_inst = typed_primitive_inst<lstm_dynamic_input>;

}  // namespace cldnn
