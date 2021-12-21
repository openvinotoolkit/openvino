// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "intel_gpu/primitives/mutable_data.hpp"
#include "primitive_inst.h"

#include <string>
#include <memory>

namespace cldnn {

template <>
struct typed_program_node<mutable_data> : public typed_program_node_base<mutable_data> {
    using parent = typed_program_node_base<mutable_data>;

    typed_program_node(const std::shared_ptr<mutable_data> prim, program& prog);

    memory& get_attached_memory() const { return *mem; }
    memory::ptr get_attached_memory_ptr() const { return mem; }
    void attach_memory(memory::ptr new_mem, bool invalidate_users_if_changed = true);

    program_node& input(size_t idx = 0) const { return get_dependency(idx); }

private:
    memory::ptr mem;
};

using mutable_data_node = typed_program_node<mutable_data>;

template <>
class typed_primitive_inst<mutable_data> : public typed_primitive_inst_base<mutable_data> {
    using parent = typed_primitive_inst_base<mutable_data>;

public:
    static layout calc_output_layout(mutable_data_node const& node) { return node.get_attached_memory().get_layout(); }
    static std::string to_string(mutable_data_node const& node);

    typed_primitive_inst(network& network, mutable_data_node const& node);
    void set_output_memory(memory::ptr mem, bool check = true) override;
};

using mutable_data_inst = typed_primitive_inst<mutable_data>;

}  // namespace cldnn
