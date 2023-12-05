// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/lstm_dynamic_timeloop.hpp"
#include "primitive_inst.h"

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
    typed_program_node(std::shared_ptr<primitive> prim, program& prog)
        : parent(std::move(prim), prog) {
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
    int32_t direction() const { return recurrent().get_output_layout().feature(); }
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
    using parent::parent;

public:
    static layout calc_output_layout(lstm_dynamic_timeloop_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(lstm_dynamic_timeloop_node const& node);

public:
    typed_primitive_inst(network& network, lstm_dynamic_timeloop_node const& node);

    memory::ptr dyn_length_memory() const { return get_dependency_memory("dyn_length"); }
    memory::ptr recurrent_memory() const { return get_dependency_memory("recurrent"); }
    memory::ptr last_hidden_output_memory() const { return get_dependency_memory("last_hidden_output"); }
    memory::ptr last_cell_output_memory() const { return get_dependency_memory("last_cell_output"); }
    memory::ptr initial_hidden_memory() const { return get_dependency_memory("initial_hidden"); }
    memory::ptr initial_cell_memory() const { return get_dependency_memory("initial_cell"); }

    bool dyn_length_term() const { return node->dyn_length_term(); }
    bool initial_hidden_term() const { return node->initial_hidden_term(); }
    bool initial_cell_term() const { return node->initial_cell_term(); }
    bool last_hidden_output_term() const { return node->last_hidden_output_term(); }
    bool last_cell_output_term() const { return node->last_cell_output_term(); }

private:
    memory::ptr get_dependency_memory(std::string val) const { return dep_memory_ptr(node->get_dependency_idx(val)); }
};

using lstm_dynamic_timeloop_inst = typed_primitive_inst<lstm_dynamic_timeloop>;

}  // namespace cldnn
