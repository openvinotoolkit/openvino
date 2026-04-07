// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "data_inst.h"
#include "json_object.h"
#include "primitive_type_base.h"

#include "intel_gpu/runtime/memory.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/util/parallel_io.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(data)

namespace {
memory::ptr attach_or_copy_data(network& network, memory::ptr mem) {
    auto& engine = network.get_engine();
    if (mem->is_allocated_by(engine))
        return mem;

    memory::ptr result = engine.allocate_memory(mem->get_layout(), false);
    mem_lock<char, mem_lock_type::read> src(mem, network.get_stream());
    mem_lock<char, mem_lock_type::write> dst(result, network.get_stream());
    const size_t data_size = src.size();
    if (data_size >= ov::util::default_parallel_io_threshold) {
        char* src_ptr = src.data();
        char* dst_ptr = dst.data();
        const size_t max_threads = static_cast<size_t>(parallel_get_max_threads());
        const size_t num_chunks = std::max(size_t{1}, std::min(data_size / ov::util::default_parallel_io_min_chunk, max_threads));
        const size_t chunk_size = (data_size + num_chunks - 1) / num_chunks;
        ov::parallel_for(num_chunks, [src_ptr, dst_ptr, chunk_size, data_size, num_chunks](size_t i) {
            const size_t offset = i * chunk_size;
            const size_t copy_size = (i + 1 == num_chunks) ? (data_size - offset) : chunk_size;
            std::memcpy(dst_ptr + offset, src_ptr + offset, copy_size);
        });
    } else {
        std::copy(src.begin(), src.end(), dst.begin());
    }
    return result;
}
}  // namespace

data_node::typed_program_node(const std::shared_ptr<data> dprim, program& prog)
    : parent(dprim, prog), mem(dprim->mem) {
    constant = true;
    can_share_buffer(false);
    recalc_output_layout(false);
}

void data_node::attach_memory(memory::ptr new_mem, bool invalidate_users_if_changed) {
    mem = new_mem;
    recalc_output_layout(invalidate_users_if_changed);
}

std::string data_inst::to_string(data_node const& node) {
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    node_info->dump(primitive_description);
    return primitive_description.str();
}

data_inst::typed_primitive_inst(network& network, data_node const& node)
    : parent(network, node, attach_or_copy_data(network, node.get_attached_memory_ptr())) {}

}  // namespace cldnn
