// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "intel_gpu/primitives/condition.hpp"
#include "primitive_inst.h"

#include <string>
#include <memory>

namespace cldnn {
namespace details {}

template <>
struct typed_program_node<condition> : public typed_program_node_base<condition> {
private:
    using parent = typed_program_node_base<condition>;

    class branch {
    public:
        explicit branch(const topology& tpl) : _topology(tpl) {}

        void set(const program_node& node) {
            add_or_change_input_layout(node);
            _program = program::build_program(node.get_program().get_engine(),
                                              _topology,
                                              node.get_program().get_options(),
                                              true);  // rebuild program
        }
        program::ptr get() const { return _program; }

    private:
        topology _topology;
        program::ptr _program = nullptr;

        void add_or_change_input_layout(const program_node& node) {
            auto layout = node.get_dependency(0).get_output_layout();
            auto input_id = node.as<condition>().result_id();
            if (_topology.get_primitives().count(input_id) == 0) {
                _topology.add_primitive(std::make_shared<input_layout>(input_id, layout));
                for (auto& prim : _topology.get_primitives()) {
                    for (auto& inp : prim.second->input) {
                        if (inp == node.id())
                            inp = input_id;
                    }
                }
            } else {
                _topology.change_input_layout(input_id, layout);
            }
        }
    };

public:
    using parent::parent;

    typed_program_node(std::shared_ptr<primitive> prim, program& prog)
        : parent(prim, prog),
          _branch_true(this->get_primitive()->topology_true),
          _branch_false(this->get_primitive()->topology_false) {}

    program_node& input() const { return get_dependency(0); }
    program_node& compare() const { return get_dependency(1); }
    cond_functions func() const { return get_primitive()->function; }
    tensor offset() const { return get_primitive()->offset; }
    void set_branches() const {
        _branch_true.set(*this);
        _branch_false.set(*this);
    }
    program::ptr get_branch_true() const { return _branch_true.get(); }
    program::ptr get_branch_false() const { return _branch_false.get(); }
    primitive_id result_id() const { return id() + ":result"; }

private:
    mutable branch _branch_true;
    mutable branch _branch_false;
};

using condition_node = typed_program_node<condition>;

template <>
class typed_primitive_inst<condition> : public typed_primitive_inst_base<condition> {
    using parent = typed_primitive_inst_base<condition>;

public:
    static layout calc_output_layout(condition_node const& node);
    static std::string to_string(condition_node const& node);
    typed_primitive_inst(network& network, condition_node const& node);

    memory::ptr input_memory_ptr() const { return dep_memory_ptr(0); }
    memory::ptr compare_memory_ptr() const { return dep_memory_ptr(1); }
    memory& input_memory() const { return dep_memory(0); }
    memory& compare_memory() const { return dep_memory(1); }
    network::ptr get_net_true() const { return _net_true; }
    network::ptr get_net_false() const { return _net_false; }
    primitive_id result_id() const { return node.result_id(); }

private:
    network::ptr _net_true;
    network::ptr _net_false;
};

using condition_inst = typed_primitive_inst<condition>;
}  // namespace cldnn
