// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <map>
#include <string>

#include "data_inst.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include "moe_3gemm_fused_inst.h"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "primitive_type_base.h"
#include "program_node.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(moe_3gemm_fused_compressed)

namespace {
void validate_partial_buffer(const data_node& buffer, const std::vector<std::pair<const program_node*, size_t>>& consumers) {
    const auto mem = buffer.get_attached_memory_ptr();
    const auto& shape = mem->get_layout().get_shape();
    const auto resident_experts = buffer.get_primitive()->get_resident_expert_num();
    OPENVINO_ASSERT(!buffer.is_output() && !consumers.empty(),
                    "Invalid partial OTD buffer ",
                    buffer.id(),
                    ": buffer must only be consumed by offloaded MoE routed inputs");
    OPENVINO_ASSERT(!shape.empty() && resident_experts > 0 && resident_experts <= shape[0],
                    "Invalid resident expert count for partial OTD buffer ",
                    buffer.id());
    const auto expert_bytes = mem->get_layout().bytes_count() / shape[0];
    const auto tracker = mem->get_mem_tracker();
    OPENVINO_ASSERT(tracker && expert_bytes > 0 && tracker->size() / expert_bytes >= resident_experts,
                    "Insufficient resident capacity for partial OTD buffer ",
                    buffer.id());

    for (const auto& consumer : consumers) {
        const auto* user = consumer.first;
        const auto input_idx = consumer.second;
        OPENVINO_ASSERT(user->is_type<moe_3gemm_fused_compressed>(), "Invalid consumer of partial OTD buffer ", buffer.id(), ": ", user->id());
        const auto& moe = user->as<moe_3gemm_fused_compressed>();
        moe.validate_partial_buffer_input(mem->get_layout(), resident_experts, input_idx, buffer.id());
    }
}
}  // namespace

void validate_moe_partial_buffers(const program& prog) {
    std::map<const program_node*, std::vector<std::pair<const program_node*, size_t>>> partial_buffer_consumers;
    // Derive consumers from dependencies; cached user lists are not a trusted source.
    for (const auto& entry : prog.nodes_map) {
        const auto& consumer = *entry.second;
        const auto& dependencies = consumer.get_dependencies();
        for (size_t input_idx = 0; input_idx < dependencies.size(); ++input_idx) {
            const auto* dependency = dependencies[input_idx].first;
            if (dependency->is_type<data>() && dependency->as<data>().get_primitive()->skip_device_transfer()) {
                partial_buffer_consumers[dependency].emplace_back(&consumer, input_idx);
            }
        }
    }
    for (const auto& entry : prog.nodes_map) {
        if (entry.second->is_type<data>() && entry.second->as<data>().get_primitive()->skip_device_transfer()) {
            OPENVINO_ASSERT(std::find(prog.outputs.begin(), prog.outputs.end(), entry.second.get()) == prog.outputs.end(),
                            "Partial OTD buffer cannot be a graph output: ",
                            entry.first);
            validate_partial_buffer(entry.second->as<data>(), partial_buffer_consumers[entry.second.get()]);
        }
    }
}

void moe_node::validate_partial_buffer_input(const layout& buffer_layout, size_t resident_experts, size_t input_idx, const primitive_id& buffer_id) const {
    const auto moe = get_primitive();
    const auto& shape = buffer_layout.get_shape();
    OPENVINO_ASSERT(!shape.empty() && moe->_config.num_expert > 0 && static_cast<size_t>(moe->_config.num_expert) == shape[0] && moe->_otd.lru_expert_num > 0 &&
                        moe->_otd.lru_expert_num <= resident_experts &&
                        moe->_otd.weight_bin_offsets.size() == moe_3gemm_fused_compressed::serialized_weight_offset_count,
                    "Incompatible offloaded MoE consumer of partial OTD buffer ",
                    buffer_id,
                    ": ",
                    id());
    OPENVINO_ASSERT(input_idx >= static_cast<size_t>(moe_3gemm_fused_compressed::input_index::weight_0) && input_idx < moe_3gemm_fused_compressed::input_count,
                    "Invalid input index for partial OTD buffer ",
                    buffer_id,
                    ": ",
                    input_idx);
}

/*
    Calc_output_layout method is called only when output layout is invalidated.
    It means, that it is called when:
    1) It has never been called.
    2) Dependency has changed output layout.
    In this both cases, we need to recalc branch_true and branch_false.
    !* We can be sure, that this method was called AT LEAST once during graph compilation.*!
*/
layout moe_inst::calc_output_layout(const moe_node& /* node */, const kernel_impl_params& impl_param) {
    return impl_param.input_layouts[0];
}

template <typename ShapeType>
std::vector<layout> moe_inst::calc_output_layouts(const moe_node& /* node */, const kernel_impl_params& impl_param) {
    return {impl_param.input_layouts[0]};
}

template std::vector<layout> moe_inst::calc_output_layouts<ov::PartialShape>(const moe_node& node, const kernel_impl_params& impl_param);

std::string moe_inst::to_string(const moe_node& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    json_composite moe_info;

    node_info->add("moe info", moe_info);

    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

/*
moe primitive is reusing memory with the input.
*/
moe_inst::typed_primitive_inst(network& network, const moe_node& node) : parent(network, node) {}

}  // namespace cldnn
