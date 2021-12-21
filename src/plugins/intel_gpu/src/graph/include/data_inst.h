// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "intel_gpu/primitives/data.hpp"
#include "primitive_inst.h"

#include <string>
#include <memory>

namespace cldnn {

template <>
struct typed_program_node<data> : public typed_program_node_base<data> {
    using parent = typed_program_node_base<data>;

    typed_program_node(const std::shared_ptr<data> prim, program& prog);

    memory& get_attached_memory(int32_t idx = 0) const { return *mems[idx]; }
    memory::ptr get_attached_memory_ptr(int32_t idx = 0) const { return mems[idx]; }
    std::vector<memory::ptr> get_attached_memory_ptrs() const { return mems; }
    void attach_memory(memory::ptr new_mem, bool invalidate_users_if_changed = true, int32_t idx = 0);

private:
    std::vector<memory::ptr> mems;
};

using data_node = typed_program_node<data>;

template <>
class typed_primitive_inst<data> : public typed_primitive_inst_base<data> {
    using parent = typed_primitive_inst_base<data>;

public:
    static layout calc_output_layout(data_node const& node) { return node.get_attached_memory().get_layout(); }
    static std::string to_string(data_node const& node);

public:
    typed_primitive_inst(network& network, data_node const& node);
};

using data_inst = typed_primitive_inst<data>;

}  // namespace cldnn
