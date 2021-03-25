// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "api/mutable_data.hpp"
#include "primitive_inst.h"
#include <string>
#include <memory>

namespace cldnn {

template <>
struct typed_program_node<mutable_data> : public typed_program_node_base<mutable_data> {
    using parent = typed_program_node_base<mutable_data>;

    typed_program_node(const std::shared_ptr<mutable_data> prim, program_impl& prog);

    memory_impl& get_attached_memory() const { return *mem; }
    memory_impl::ptr get_attached_memory_ptr() const { return mem; }
    void attach_memory(memory_impl& new_mem, bool invalidate_users_if_changed = true);

    program_node& input(size_t idx = 0) const { return get_dependency(idx); }

private:
    memory_impl::ptr mem;

    void fill_memory();
    void fill_memory_xavier();
    void fill_memory_constant(float value);
};

using mutable_data_node = typed_program_node<mutable_data>;

template <>
class typed_primitive_inst<mutable_data> : public typed_primitive_inst_base<mutable_data> {
    using parent = typed_primitive_inst_base<mutable_data>;

public:
    static layout calc_output_layout(mutable_data_node const& node) { return node.get_attached_memory().get_layout(); }
    static std::string to_string(mutable_data_node const& node);

public:
    typed_primitive_inst(network_impl& network, mutable_data_node const& node);
};

using mutable_data_inst = typed_primitive_inst<mutable_data>;

}  // namespace cldnn
