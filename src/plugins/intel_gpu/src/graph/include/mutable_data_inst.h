// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
    void replace_memory(memory::ptr new_mem, bool invalidate_users_if_changed = false);

    program_node& input(size_t idx = 0) const { return get_dependency(idx); }

private:
    memory::ptr mem;
};

using mutable_data_node = typed_program_node<mutable_data>;

template <>
class typed_primitive_inst<mutable_data> : public typed_primitive_inst_base<mutable_data> {
    using parent = typed_primitive_inst_base<mutable_data>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(mutable_data_node const& node, const kernel_impl_params& impl_param) {
        return { node.get_attached_memory().get_layout() };
    }

    static layout calc_output_layout(mutable_data_node const& node, kernel_impl_params const& impl_param) {
        return node.get_attached_memory().get_layout();
    }

    static std::string to_string(mutable_data_node const& node);

    typed_primitive_inst(network& network, mutable_data_node const& node);
    event::ptr set_output_memory(memory::ptr mem, bool check = true, size_t idx = 0) override;
    const std::list<primitive_id>& get_user_ids() const { return _user_ids; }

private:
    std::list<primitive_id> _user_ids;
};

using mutable_data_inst = typed_primitive_inst<mutable_data>;

}  // namespace cldnn
