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
#include "api_extension/lstm_dynamic_timeloop.hpp"
#include "primitive_inst.h"
#include "error_handler.h"
#include <memory>
#include <string>
#include <vector>

namespace cldnn {

template <>
struct typed_program_node<lstm_dynamic_timeloop> : public typed_program_node_base<lstm_dynamic_timeloop> {
    using parent = typed_program_node_base<lstm_dynamic_timeloop>;

private:
    std::vector<std::string> _param_list;
    program_node& get_dependency_by_name(std::string val) const;
    void init_params_list();
    inline size_t get_param_list_index(const std::string& dependency_tag) const {
        return static_cast<size_t>(std::distance(_param_list.begin(), std::find_if(
        _param_list.begin(), _param_list.end(), [&](const std::string& tag) { return tag == dependency_tag; })));
    }

public:
    typed_program_node(std::shared_ptr<primitive> prim, program_impl& prog)
        : parent(prim, prog) {
        init_params_list();
        can_share_buffer(false);
    }

    void reverse_optional_outputs_connections();
    size_t get_dependency_idx(std::string val) const;

    program_node& input() const { return get_dependency_by_name("input"); }
    program_node& dyn_length() const { return get_dependency_by_name("dyn_length"); }
    program_node& recurrent() const { return get_dependency_by_name("recurrent"); }
    program_node& last_hidden_state() const { return get_dependency_by_name("last_hidden_output"); }
    program_node& last_cell_state() const { return get_dependency_by_name("last_cell_output"); }
    program_node& initial_hidden() const { return get_dependency_by_name("initial_hidden"); }
    program_node& initial_cell() const { return get_dependency_by_name("initial_cell"); }

    float clip() const { return get_primitive()->clip; }
    int32_t direction() const { return recurrent().get_output_layout().size.feature[0]; }
    bool input_forget() const { return get_primitive()->input_forget; }
    bool dyn_length_term() const { return !get_primitive()->dyn_length.empty(); }
    bool recurrent_term() const { return !get_primitive()->recurrent.empty(); }
    bool initial_hidden_term() const { return !get_primitive()->initial_hidden.empty(); }
    bool initial_cell_term() const { return !get_primitive()->initial_cell.empty(); }
    bool last_hidden_output_term() const { return !get_primitive()->last_hidden_state.empty(); }
    bool last_cell_output_term() const { return !get_primitive()->last_cell_state.empty(); }
};

using lstm_dynamic_timeloop_node = typed_program_node<lstm_dynamic_timeloop>;

template <>
class typed_primitive_inst<lstm_dynamic_timeloop> : public typed_primitive_inst_base<lstm_dynamic_timeloop> {
    using parent = typed_primitive_inst_base<lstm_dynamic_timeloop>;

public:
    static layout calc_output_layout(lstm_dynamic_timeloop_node const& node);
    static std::string to_string(lstm_dynamic_timeloop_node const& node);

public:
    typed_primitive_inst(network_impl& network, lstm_dynamic_timeloop_node const& node);

    memory_impl& dyn_length_memory() const { return get_dependency_memory("dyn_length"); }
    memory_impl& recurrent_memory() const { return get_dependency_memory("recurrent"); }
    memory_impl& last_hidden_output_memory() const { return get_dependency_memory("last_hidden_output"); }
    memory_impl& last_cell_output_memory() const { return get_dependency_memory("last_cell_output"); }
    memory_impl& initial_hidden_memory() const { return get_dependency_memory("initial_hidden"); }
    memory_impl& initial_cell_memory() const { return get_dependency_memory("initial_cell"); }

    bool dyn_length_term() const { return node.dyn_length_term(); }
    bool initial_hidden_term() const { return node.initial_hidden_term(); }
    bool initial_cell_term() const { return node.initial_cell_term(); }
    bool last_hidden_output_term() const { return node.last_hidden_output_term(); }
    bool last_cell_output_term() const { return node.last_cell_output_term(); }

private:
    memory_impl& get_dependency_memory(std::string val) const { return dep_memory(node.get_dependency_idx(val)); }
};

using lstm_dynamic_timeloop_inst = typed_primitive_inst<lstm_dynamic_timeloop>;

}  // namespace cldnn
