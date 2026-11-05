// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <memory>
#include <string>

#include "common_utils/parallel_memcpy.hpp"
#include "data_inst.h"
#include "intel_gpu/runtime/memory.hpp"
#include "json_object.h"
#include "openvino/util/parallel_io.hpp"
#include "primitive_type_base.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(data)

namespace {
memory::ptr attach_or_copy_data(network& network, memory::ptr mem) {
    auto& engine = network.get_engine();
    if (mem->is_allocated_by(engine)) {
        return mem;
    }

    memory::ptr result = engine.allocate_memory(mem->get_layout(), false);
    mem_lock<char, mem_lock_type::read> src(mem, network.get_stream());
    mem_lock<char, mem_lock_type::write> dst(result, network.get_stream());
    ov::intel_gpu::parallel_memcpy(dst.data(), src.data(), src.size());
    return result;
}
}  // namespace

data_node::typed_program_node(const std::shared_ptr<data> dprim, program& prog) : parent(dprim, prog), mem(dprim->mem) {
    constant = true;
    can_share_buffer(false);
    recalc_output_layout(false);
}

void data_node::attach_memory(memory::ptr new_mem, bool invalidate_users_if_changed) {
    mem = new_mem;
    recalc_output_layout(invalidate_users_if_changed);
}

std::string data_inst::to_string(const data_node& node) {
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    node_info->dump(primitive_description);
    return primitive_description.str();
}

data_inst::typed_primitive_inst(network& network, const data_node& node)
    : parent(network, node, attach_or_copy_data(network, node.get_attached_memory_ptr())) {}

}  // namespace cldnn
