// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "data_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/memory.hpp"

#include "json_object.h"
#include <string>
#include <memory>
#include <algorithm>

namespace cldnn {
primitive_type_id data::type_id() {
    static primitive_type_base<data> instance;
    return &instance;
}

namespace {
std::vector<memory::ptr> attach_or_copy_data(network& network, std::vector<memory::ptr> mems) {
    auto& engine = network.get_engine();
    std::vector<memory::ptr> result_mems;
    for (auto mem : mems) {
        if (mem->is_allocated_by(engine)) {
            result_mems.push_back(mem);
            continue;
        }

        memory::ptr result = engine.allocate_memory(mem->get_layout(), false);
        mem_lock<char, mem_lock_type::read> src(mem, network.get_stream());
        mem_lock<char, mem_lock_type::write> dst(result, network.get_stream());
        std::copy(src.begin(), src.end(), dst.begin());
        result_mems.push_back(result);
    }
    return result_mems;
}
}  // namespace

data_node::typed_program_node(const std::shared_ptr<data> dprim, program& prog)
    : parent(dprim, prog), mems(dprim->mems) {
    constant = true;
    can_share_buffer(false);
    recalc_output_layouts(false);
}

void data_node::attach_memory(memory::ptr new_mem, bool invalidate_users_if_changed, int32_t idx) {
    // TODO(taylor): multiple outputs
    mems[idx] = new_mem;
    recalc_output_layouts(invalidate_users_if_changed);
}

std::string data_inst::to_string(data_node const& node) {
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    node_info->dump(primitive_description);
    return primitive_description.str();
}

data_inst::typed_primitive_inst(network& network, data_node const& node)
    : parent(network, node, attach_or_copy_data(network, node.get_attached_memory_ptrs())) {}

}  // namespace cldnn
