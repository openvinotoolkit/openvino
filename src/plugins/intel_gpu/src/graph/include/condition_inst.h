// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
    explicit branch(const cldnn::condition::branch_info& data) : _data(data) {}

    void set(const program_node& node) {
        add_or_change_input_layout(node);
        if (_program == nullptr) {
            std::cout << "[START][" << node.id();
            std::cout << "][" << _data.tags << "] Build inner program ....... " << std::endl;
            _program = program::build_program(node.get_program().get_engine(),
                                                *_data.topology_ptr.get(),
                                                node.get_program().get_config(),
                                                true);  // rebuild program
            std::cout << "[END..] Build inner program ....... " << std::endl;
        }
    }
    program::ptr get() const { return _program; }

private:
    cldnn::condition::branch_info _data;
    program::ptr _program = nullptr;

    void add_or_change_input_layout(const program_node& node) {
        for (auto& p_iter : node.get_dependencies()) {
            auto pid = p_iter.first->id();
            auto iter = _data.input_map.find(pid);
            if (iter != _data.input_map.end()) {
                auto out_layout = p_iter.first->get_output_layout();
                _data.topology_ptr->change_input_layout(iter->second.second, out_layout);
            }
        }
    }
};

public:
    using parent::parent;

    typed_program_node(std::shared_ptr<primitive> prim, program& prog)
        : parent(prim, prog),
          _branch_true(this->get_primitive()->branch_true),
          _branch_false(this->get_primitive()->branch_false) {}

    program_node& input() const { return get_dependency(0); }
    cond_functions func() const { return get_primitive()->function; }
    tensor offset() const { return get_primitive()->offset; }
    void set_branches() const {
        _branch_true.set(*this);
        _branch_false.set(*this);
    }
    program::ptr get_branch_true() const { return _branch_true.get(); }
    program::ptr get_branch_false() const { return _branch_false.get(); }

private:
    mutable branch _branch_true;
    mutable branch _branch_false;
};

using condition_node = typed_program_node<condition>;

template <>
class typed_primitive_inst<condition> : public typed_primitive_inst_base<condition> {
    using parent = typed_primitive_inst_base<condition>;
    using parent::parent;

public:
    static layout calc_output_layout(condition_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(condition_node const& node);
    typed_primitive_inst(network& network, condition_node const& node);

    // TODO: When memory is allocated for dep input
    memory::ptr compare_memory_ptr() const { return dep_memory_ptr(0); }
    network::ptr get_net_true() const { return _net_true; }
    network::ptr get_net_false() const { return _net_false; }
    network::ptr get_networks(bool is_net_true);

private:
    network::ptr _net_true;
    network::ptr _net_false;
};

using condition_inst = typed_primitive_inst<condition>;
}  // namespace cldnn
