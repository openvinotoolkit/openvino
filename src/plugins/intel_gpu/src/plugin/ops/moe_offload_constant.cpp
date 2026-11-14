// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_offload_constant.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <unordered_set>
#include <vector>

#include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"
#include "openvino/core/weight_sharing_util.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"

namespace ov::intel_gpu {

// Input indices 3..11 are routed expert weights/scales/zps (WEIGHT_0..ZP_2).
// Input indices 12..21 are shared expert weights (SHARED_GATE_WEIGHT..SHARED_GATE_GATE_WEIGHT).
// See MOE3GemmInputIndex in moe_3gemm_base.hpp for the authoritative enum.
static constexpr size_t ROUTED_INPUT_START = 3;
static constexpr size_t ROUTED_INPUT_END = 11;
static constexpr size_t SHARED_INPUT_START = 12;
static constexpr size_t SHARED_INPUT_END = 21;

bool is_otd_capable_moe(const ov::Node* node) {
    if (const auto* moe = ov::as_type<const ov::op::internal::MOECompressed>(node)) {
        return moe->get_config().expert_type == ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
    }
    return false;
}

MoEConstantRole get_moe_constant_role(const std::shared_ptr<ov::op::v0::Constant>& op) {
    const auto target_inputs = op->get_output_target_inputs(0);
    bool has_moe_consumer = false;
    bool has_non_otd_moe = false;
    bool has_routed = false;
    bool has_shared = false;
    bool has_other_consumer = false;

    for (const auto& input : target_inputs) {
        const auto* node = input.get_node();
        if (const auto* moe = ov::as_type<const ov::op::internal::MOECompressed>(node)) {
            has_moe_consumer = true;
            if (moe->get_config().expert_type != ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU) {
                has_non_otd_moe = true;
            } else {
                const auto idx = input.get_index();
                if (idx >= ROUTED_INPUT_START && idx <= ROUTED_INPUT_END) {
                    has_routed = true;
                } else if (idx >= SHARED_INPUT_START && idx <= SHARED_INPUT_END) {
                    has_shared = true;
                } else {
                    has_other_consumer = true;
                }
            }
        } else {
            has_other_consumer = true;
        }
    }

    if (!has_moe_consumer) {
        return MoEConstantRole::NotMoE;
    }
    if (has_non_otd_moe) {
        return MoEConstantRole::NotOffloadableMoE;
    }
    if (has_shared) {
        return MoEConstantRole::AlwaysResidentSharedExpert;
    }
    if (has_routed) {
        if (has_other_consumer) {
            return MoEConstantRole::NotOffloadableRoutedExpert;
        }
        return MoEConstantRole::OffloadableRoutedExpert;
    }
    return MoEConstantRole::NotOffloadableMoE;
}

bool is_moe_otd_eligible(const ov::Node* node) {
    std::vector<const ov::Node*> pending{node};
    std::unordered_set<const ov::Node*> visited;
    while (!pending.empty()) {
        const auto* current = pending.back();
        pending.pop_back();
        if (!visited.insert(current).second)
            continue;
        if (!is_otd_capable_moe(current) || current->get_input_size() <= ROUTED_INPUT_END)
            return false;
        for (size_t index = ROUTED_INPUT_START; index <= ROUTED_INPUT_END; ++index) {
            auto constant = ov::as_type_ptr<ov::op::v0::Constant>(current->input_value(index).get_node_shared_ptr());
            if (!constant || get_moe_constant_role(constant) != MoEConstantRole::OffloadableRoutedExpert)
                return false;
            // Propagate resident fallback across MoEs sharing routed constants.
            for (const auto& consumer : constant->get_output_target_inputs(0)) {
                pending.push_back(consumer.get_node());
            }
        }
    }
    return true;
}

bool is_moe_constant_offloadable(const std::shared_ptr<ov::op::v0::Constant>& op) {
    if (get_moe_constant_role(op) != MoEConstantRole::OffloadableRoutedExpert)
        return false;
    const auto consumers = op->get_output_target_inputs(0);
    return std::all_of(consumers.begin(), consumers.end(), [](const ov::Input<ov::Node>& consumer) {
        return is_moe_otd_eligible(consumer.get_node());
    });
}

bool is_moe_related_constant(const std::shared_ptr<ov::op::v0::Constant>& op) {
    return get_moe_constant_role(op) != MoEConstantRole::NotMoE;
}

namespace {
using moe_input_index = cldnn::moe_3gemm_fused_compressed::input_index;
constexpr std::array<size_t, 3> routed_weight_inputs = {static_cast<size_t>(moe_input_index::weight_0),
                                                        static_cast<size_t>(moe_input_index::weight_1),
                                                        static_cast<size_t>(moe_input_index::weight_2)};

bool is_routed_weight_input(size_t index) {
    return std::find(routed_weight_inputs.begin(), routed_weight_inputs.end(), index) != routed_weight_inputs.end();
}

// Offloaded experts are streamed from the weights file as stored, so the graph must consume the file bytes unchanged:
// a constant whose data was converted, transposed or otherwise recomputed by the transformations has no file descriptor
bool is_weights_file_view(const ov::Output<ov::Node>& input, size_t num_expert) {
    const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(input.get_node_shared_ptr());
    if (!constant || !ov::weight_sharing::Extension::get_constant_source_buffer(*constant)) {
        return false;
    }
    const auto& shape = constant->get_shape();
    // Each expert slice must start on a byte boundary (sub-byte weights)
    return !shape.empty() && shape[0] == num_expert && (ov::shape_size(shape) * constant->get_element_type().bitwidth()) % (8 * num_expert) == 0;
}

// Slot count of the offloaded layer that consumes the constant as a routed weight, 0 for any other constant
size_t get_routed_weight_slot_count(const std::shared_ptr<ov::op::v0::Constant>& op, size_t offload_ratio) {
    for (const auto& input : op->get_output_target_inputs(0)) {
        const auto* moe = ov::as_type<const ov::op::internal::MOECompressed>(input.get_node());
        if (moe && is_routed_weight_input(input.get_index())) {
            return get_moe_offload_slot_count(*moe, offload_ratio);
        }
    }
    return 0;
}
}  // namespace

size_t get_moe_offload_slot_count(const ov::op::internal::MOECompressed& moe, size_t offload_ratio) {
    const auto& moe_config = moe.get_config();
    // ratio=0 keeps all experts resident
    if (offload_ratio == 0 || moe_config.expert_type != ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU || moe_config.num_expert == 0 ||
        moe.get_input_size() <= routed_weight_inputs.back() || !is_moe_otd_eligible(&moe)) {
        return 0;
    }
    const auto num_expert = static_cast<size_t>(moe_config.num_expert);
    for (const auto index : routed_weight_inputs) {
        if (!is_weights_file_view(moe.input_value(index), num_expert)) {
            return 0;
        }
    }
    // offload_ratio is the % on disk; GPU-resident experts = max(1, total * (100 - ratio) / 100)
    return std::max<size_t>(1, num_expert * (100 - offload_ratio) / 100);
}

uint64_t get_model_resident_constant_bytes(const ov::Model& model, size_t offload_ratio) {
    OPENVINO_ASSERT(offload_ratio <= 100, "OFFLOAD_RATIO must be in the range [0, 100]");

    uint64_t resident_bytes = 0;
    bool has_moe = false;
    std::unordered_set<const ov::Node*> visited;
    std::function<void(const ov::Model&)> collect_resident_bytes = [&](const ov::Model& current_model) {
        for (const auto& node : current_model.get_ops()) {
            has_moe = has_moe || is_otd_capable_moe(node.get());
            if (auto subgraph_op = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(node)) {
                for (const auto& subgraph : subgraph_op->get_functions())
                    collect_resident_bytes(*subgraph);
            }

            auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
            if (!constant || !visited.insert(constant.get()).second)
                continue;

            const auto role = get_moe_constant_role(constant);
            if (role == MoEConstantRole::NotMoE)
                continue;

            uint64_t constant_resident_bytes = constant->get_byte_size();
            if (const size_t slot_count = get_routed_weight_slot_count(constant, offload_ratio)) {
                const auto& shape = constant->get_shape();
                if (shape.empty() || shape[0] == 0)
                    continue;

                const size_t resident_experts = std::min<size_t>(shape[0], slot_count);
                const uint64_t bytes_per_expert =
                    constant_resident_bytes / shape[0] + (constant_resident_bytes % shape[0] != 0 ? 1 : 0);
                constant_resident_bytes = bytes_per_expert * resident_experts;
            }

            OPENVINO_ASSERT(constant_resident_bytes <= std::numeric_limits<uint64_t>::max() - resident_bytes, "MoE resident weight size overflows uint64_t");
            resident_bytes += constant_resident_bytes;
        }
    };

    collect_resident_bytes(model);
    return has_moe ? resident_bytes : 0;
}

void validate_model_resident_constant_memory(const ov::Model& model, size_t offload_ratio, uint64_t device_memory_bytes) {
    if (offload_ratio == 0 || device_memory_bytes == 0)
        return;

    const uint64_t resident_bytes = get_model_resident_constant_bytes(model, offload_ratio);
    OPENVINO_ASSERT(resident_bytes <= device_memory_bytes,
                    "[GPU] Resident model constants require at least ",
                    resident_bytes,
                    " bytes, exceeding the device global memory size of ",
                    device_memory_bytes,
                    " bytes. Increase OFFLOAD_RATIO or use a GPU with more memory.");
}

PartialUploadLogState& get_partial_upload_log_state() {
    static PartialUploadLogState state;
    return state;
}

PartialUploadDesc try_prepare_partial_upload(cldnn::engine& engine,
                                             const ExecutionConfig& config,
                                             const std::shared_ptr<ov::op::v0::Constant>& op,
                                             const ov::Shape& const_shape,
                                             cldnn::data_types out_dtype,
                                             const cldnn::format& const_format,
                                             const cldnn::layout& const_layout) {
    PartialUploadDesc desc;

    // Only the routed expert weights of an offloaded layer are partially uploaded:
    // scales, zero points and shared experts stay fully resident.
    const size_t resident_expert_num = get_routed_weight_slot_count(op, config.get_offload_ratio());
    if (resident_expert_num == 0 || const_layout.bytes_count() == 0 || const_shape.empty() || const_shape[0] == 0) {
        return desc;
    }

    desc.enabled = true;
    desc.upload_shape = const_shape;
    desc.upload_shape[0] = std::min<size_t>(const_shape[0], resident_expert_num);

    auto upload_layout = cldnn::layout(desc.upload_shape, out_dtype, const_format);
    auto upload_mem = engine.allocate_memory(upload_layout, engine.get_preferred_memory_allocation_type(), false);
    // Reinterpret the smaller physical allocation as the full constant layout so the
    // graph sees the expected shape/layout. This is safe because:
    // 1. constant.cpp marks this data node with skip_device_transfer=true (partial_upload.enabled),
    //    so no host→device memcpy of the full size occurs.
    // 2. At runtime, OTD loads on-demand into the first `resident_expert_num` slots only.
    // 3. Weightless cache serialization uses bin_offset metadata for these constants and
    //    never reads the buffer contents via mem->buffer_ptr(). OTD provides weights_path,
    //    but does not enable weightless caching itself.
    // TODO: Support serialization of OTD partial allocations without weightless caching.
    OPENVINO_ASSERT(upload_layout.bytes_count() <= const_layout.bytes_count(),
                    "Partial upload layout (", upload_layout.bytes_count(),
                    " bytes) exceeds full constant layout (", const_layout.bytes_count(), " bytes)");
    desc.memory = engine.reinterpret_buffer(*upload_mem, const_layout);
    desc.upload_bytes = upload_layout.bytes_count();

    get_partial_upload_log_state().log(op->get_friendly_name(),
                                       desc.upload_shape[0],
                                       const_shape[0],
                                       desc.upload_bytes,
                                       const_layout.bytes_count());
    return desc;
}

}  // namespace ov::intel_gpu
