// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_offload_constant.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <unordered_set>

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

bool is_moe_related_constant(const std::shared_ptr<ov::op::v0::Constant>& op) {
    return get_moe_constant_role(op) != MoEConstantRole::NotMoE;
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
            if (role == MoEConstantRole::OffloadableRoutedExpert && offload_ratio > 0) {
                const auto& shape = constant->get_shape();
                if (shape.empty() || shape[0] == 0)
                    continue;

                const size_t resident_experts = std::max<size_t>(1, shape[0] * (100 - offload_ratio) / 100);
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

    const size_t otd_ratio = config.get_offload_ratio();
    // Only routed expert weights are partially uploaded; shared experts stay fully resident.
    // ratio=0 keeps all experts resident; positive ratios enable partial upload.
    const bool partial_moe_const_upload = otd_ratio > 0 &&
                                          get_moe_constant_role(op) == MoEConstantRole::OffloadableRoutedExpert;
    if (!partial_moe_const_upload || const_layout.bytes_count() == 0 || const_shape.empty() || const_shape[0] == 0) {
        return desc;
    }

    // otd_ratio is the % on disk; GPU-resident experts = max(1, total * (100 - ratio) / 100)
    const size_t resident_expert_num = std::max<size_t>(1, const_shape[0] * (100 - otd_ratio) / 100);

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
