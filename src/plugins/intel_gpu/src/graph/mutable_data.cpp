// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "mutable_data_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/memory.hpp"
#include "json_object.h"
#include <string>
#include <memory>
#include <algorithm>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(mutable_data)

namespace {
memory::ptr attach_or_copy_data(network& network, memory::ptr mem, bool reuse) {
    auto& engine = network.get_engine();
    auto& stream = network.get_stream();

    if (mem->is_allocated_by(engine) && reuse) {
        return mem;
    }

    memory::ptr result = engine.allocate_memory(mem->get_layout(), false);
    mem_lock<char, mem_lock_type::read> src(mem, stream);
    mem_lock<char, mem_lock_type::write> dst(result, stream);
    std::copy(src.begin(), src.end(), dst.begin());

    return result;
}
}  // namespace

mutable_data_node::typed_program_node(const std::shared_ptr<mutable_data> dprim, program& prog)
    : parent(dprim, prog), mem(dprim->mem) {
    recalc_output_layout(false);
    can_share_buffer(false);
}

void mutable_data_node::attach_memory(memory::ptr new_mem, bool invalidate_users_if_changed) {
    mem = std::move(new_mem);
    recalc_output_layout(invalidate_users_if_changed);
}

void mutable_data_node::replace_memory(memory::ptr new_mem, bool invalidate_users_if_changed) {
    mem = new_mem;
    recalc_output_layout(invalidate_users_if_changed);
}

std::string mutable_data_inst::to_string(mutable_data_node const& node) {
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    node_info->dump(primitive_description);
    return primitive_description.str();
}

event::ptr mutable_data_inst::set_output_memory(memory::ptr mem_new, bool check, size_t idx) {
    event::ptr input_ev = nullptr;
    if (_node != nullptr) {
        auto& eng = _network.get_engine();
        auto& mem_node = const_cast<program_node*>(_node)->as<mutable_data>();
        auto& mem_attached = mem_node.get_attached_memory();
        const auto& mem_orig = *_outputs[idx];
        if (!eng.is_the_same_buffer(*mem_new, mem_attached)) {
            if (_node->is_input()) {
                input_ev = mem_new->copy_from(_network.get_stream(), *_outputs[idx], false);
            }

            // re-attach mutable_data internal memory if necessary
            if (eng.is_the_same_buffer(mem_orig, mem_attached)) {
                mem_node.attach_memory(eng.reinterpret_buffer(*mem_new, mem_attached.get_layout()));
            }
        }
    }
    auto ev = primitive_inst::set_output_memory(mem_new, check);
    if (input_ev == nullptr)
        return ev;
    else
        return _network.get_stream().group_events({ev, input_ev});
}

mutable_data_inst::typed_primitive_inst(network& network, mutable_data_node const& node)
    : parent(network, node, attach_or_copy_data(network, node.get_attached_memory_ptr(), network.is_primary_stream())) {
    const auto& users = get_users();
    for (const auto& usr : users) {
        _user_ids.emplace_back(usr->id());
    }
}

}  // namespace cldnn
