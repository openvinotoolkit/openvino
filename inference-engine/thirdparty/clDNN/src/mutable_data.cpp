// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "mutable_data_inst.h"
#include "primitive_type_base.h"
#include "cldnn/runtime/memory.hpp"
#include <random>
#include "cldnn/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>
#include <memory>
#include <algorithm>

namespace cldnn {
primitive_type_id mutable_data::type_id() {
    static primitive_type_base<mutable_data> instance;
    return &instance;
}

namespace {
memory::ptr attach_or_copy_data(network_impl& network, memory::ptr mem, bool reuse) {
    auto& engine = network.get_engine();
    auto& stream = network.get_stream();

    if (mem->is_allocated_by(engine) && reuse) {
        return mem;
    }

    memory::ptr result = engine.allocate_memory(mem->get_layout(), false);
    mem_lock<char> src(mem, stream);
    mem_lock<char> dst(result, stream);
    std::copy(src.begin(), src.end(), dst.begin());

    return result;
}
}  // namespace

mutable_data_node::typed_program_node(const std::shared_ptr<mutable_data> dprim, program_impl& prog)
    : parent(dprim, prog), mem(dprim->mem) {
    recalc_output_layout(false);
    can_share_buffer(false);
}

void mutable_data_node::attach_memory(memory::ptr new_mem, bool invalidate_users_if_changed) {
    mem = new_mem;
    recalc_output_layout(invalidate_users_if_changed);
}

std::string mutable_data_inst::to_string(mutable_data_node const& node) {
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    node_info->dump(primitive_description);
    return primitive_description.str();
}

mutable_data_inst::typed_primitive_inst(network_impl& network, mutable_data_node const& node)
    : parent(network, node, attach_or_copy_data(network, node.get_attached_memory_ptr(), network.is_primary_stream())) {}

}  // namespace cldnn
