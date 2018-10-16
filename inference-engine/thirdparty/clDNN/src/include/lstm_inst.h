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
#include "api/CPP/lstm.hpp"
#include "primitive_inst.h"

namespace cldnn
{
template <>
struct typed_program_node<lstm> : public typed_program_node_base<lstm>
{
    using parent = typed_program_node_base<lstm>;

public:
    using parent::parent;

    decltype(auto) input() const { return get_dependency(0); }
    decltype(auto) weights() const { return get_dependency(1); }
    decltype(auto) recurrent() const { return get_dependency(2); }
    decltype(auto) bias() const { return get_dependency(3); }
    decltype(auto) inital_hidden() const {
        return get_dependency(bias_term() ? 4 : 3);
    }
    decltype(auto) inital_cell() const {
        // This doesn't scale. We should use a map to get the dependencies index at primitive level
        return get_dependency(bias_term() ? (initial_hidden_term() ? 5 : 4) : (initial_hidden_term() ? 4 : 2));
    }
    decltype(auto) peepholes() const { return get_dependency(6); }
    bool bias_term() const { return !get_primitive()->bias.empty(); }
    bool peepholes_term() const { return !get_primitive()->peepholes.empty(); }
    bool initial_hidden_term() const { return !get_primitive()->initial_hidden.empty(); }
    bool initial_cell_term() const { return !get_primitive()->initial_cell.empty(); }
    auto activations() const { return get_primitive()->activations; }
    auto activation_params() const { return get_primitive()->activation_params; }
};

using lstm_node = typed_program_node<lstm>;

template <>
class typed_primitive_inst<lstm> : public typed_primitive_inst_base<lstm>
{
    using parent = typed_primitive_inst_base<lstm>;

public:
    static layout calc_output_layout(lstm_node const& node);
    static std::string to_string(lstm_node const& node);

public:
    typed_primitive_inst(network_impl& network, lstm_node const& node);

    decltype(auto) weights_memory() const { return dep_memory(1); }
    decltype(auto) recurrent_memory() const { return dep_memory(2); }
    decltype(auto) bias_memory() const { return dep_memory(3); }
    decltype(auto) initial_hidden_memory() const
    {
        return dep_memory(bias_term() ? 4 : 3);
    }
    decltype(auto) initial_cell_memory() const {
        return dep_memory(bias_term() ? (initial_hidden_term() ? 5 : 4) : (initial_hidden_term() ? 4 : 2));
    }
    decltype(auto) peepholes_memory() const { return dep_memory(6); }
    bool bias_term() const { return !argument.bias.empty(); }
    bool peepholes_term() const { return !argument.peepholes.empty(); }
    bool initial_hidden_term() const { return !argument.initial_hidden.empty(); }
    bool initial_cell_term() const { return !argument.initial_cell.empty(); }
    auto activations() const { return argument.activations; }
    auto activation_params() const { return argument.activation_params; }
};

using lstm_inst = typed_primitive_inst<lstm>;

}
