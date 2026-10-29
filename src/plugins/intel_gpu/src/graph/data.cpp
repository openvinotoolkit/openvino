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
#include "moe_3gemm_fused_inst.h"
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

void data_node::validate_partial_buffer(const std::vector<std::pair<const program_node*, size_t>>& consumers) const {
    const auto desc = get_primitive();
    if (!desc->skip_device_transfer()) {
        return;
    }

    const auto& shape = mem->get_layout().get_shape();
    const auto resident_experts = desc->get_resident_expert_num();
    OPENVINO_ASSERT(!is_output() && !consumers.empty(),
                    "Invalid partial OTD buffer ", id(), ": buffer must only be consumed by offloaded MoE routed inputs");
    OPENVINO_ASSERT(!shape.empty() && resident_experts > 0 && resident_experts <= shape[0],
                    "Invalid resident expert count for partial OTD buffer ", id());
    const auto expert_bytes = mem->get_layout().bytes_count() / shape[0];
    const auto tracker = mem->get_mem_tracker();
    OPENVINO_ASSERT(tracker && expert_bytes > 0 && tracker->size() / expert_bytes >= resident_experts,
                    "Insufficient resident capacity for partial OTD buffer ", id());

    for (const auto& consumer : consumers) {
        const auto* user = consumer.first;
        const auto input_idx = consumer.second;
        OPENVINO_ASSERT(user->is_type<moe_3gemm_fused_compressed>(),
                        "Invalid consumer of partial OTD buffer ", id(), ": ", user->id());
        const auto moe = user->as<moe_3gemm_fused_compressed>().get_primitive();
        OPENVINO_ASSERT(moe->_config.num_expert > 0 && static_cast<size_t>(moe->_config.num_expert) == shape[0] &&
                            moe->_otd.lru_expert_num > 0 && moe->_otd.lru_expert_num <= resident_experts &&
                            !moe->_otd.weights_path.empty() &&
                            moe->_otd.weight_bin_offsets.size() == moe_3gemm_fused_compressed::serialized_weight_offset_count,
                        "Incompatible offloaded MoE consumer of partial OTD buffer ", id(), ": ", user->id());
        OPENVINO_ASSERT(input_idx >= static_cast<size_t>(moe_3gemm_fused_compressed::input_index::weight_0) &&
                            input_idx < moe_3gemm_fused_compressed::input_count,
                        "Invalid input index for partial OTD buffer ", id(), ": ", input_idx);
    }
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
