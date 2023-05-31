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

public:
    using parent::parent;

    typed_program_node(std::shared_ptr<primitive> prim, program& prog)
        : parent(prim, prog),
          _branch_true(this->get_primitive()->branch_true),
          _branch_false(this->get_primitive()->branch_false) {}

    program_node& input() const { return get_dependency(0); }
    program::ptr get_branch_true() const { return _branch_true.inner_program; }
    program::ptr get_branch_false() const { return _branch_false.inner_program; }

private:
    mutable condition::branch_info _branch_true;
    mutable condition::branch_info _branch_false;
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

    memory::ptr compare_memory_ptr() const { return dep_memory_ptr(0); }
    network::ptr get_net_true() const { return _net_true; }
    network::ptr get_net_false() const { return _net_false; }
    network::ptr get_inner_networks(bool is_net_true);

private:
    network::ptr _net_true;
    network::ptr _net_false;
};

using condition_inst = typed_primitive_inst<condition>;
}  // namespace cldnn
