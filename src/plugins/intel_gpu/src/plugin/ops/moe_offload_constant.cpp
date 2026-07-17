// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_offload_constant.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_set>

#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#elif defined(__linux__)
#    include <unistd.h>
#endif

namespace ov::intel_gpu {

// Input indices 3..11 are routed expert weights/scales/zps (WEIGHT_0..ZP_2).
// Input indices 12..21 are shared expert weights (SHARED_GATE_WEIGHT..SHARED_GATE_GATE_WEIGHT).
// See MOE3GemmInputIndex in moe_3gemm_base.hpp for the authoritative enum.
static constexpr size_t ROUTED_INPUT_START = 3;
static constexpr size_t ROUTED_INPUT_END = 11;
static constexpr size_t SHARED_INPUT_START = 12;
static constexpr size_t SHARED_INPUT_END = 21;

MoEConstantRole get_moe_constant_role(const std::shared_ptr<ov::op::v0::Constant>& op) {
    const auto users = op->get_output_target_inputs(0);
    for (const auto& input : users) {
        const auto* node = input.get_node();
        if (ov::is_type<ov::op::internal::MOECompressed>(node)) {
            auto idx = input.get_index();
            if (idx >= ROUTED_INPUT_START && idx <= ROUTED_INPUT_END)
                return MoEConstantRole::RoutedExpert;
            if (idx >= SHARED_INPUT_START && idx <= SHARED_INPUT_END)
                return MoEConstantRole::SharedExpert;
        }
    }
    return MoEConstantRole::NotMoE;
}

bool is_moe_related_constant(const std::shared_ptr<ov::op::v0::Constant>& op) {
    return get_moe_constant_role(op) != MoEConstantRole::NotMoE;
}

PartialUploadLogState& get_partial_upload_log_state() {
    static PartialUploadLogState state;
    return state;
}

PartialUploadDesc try_prepare_partial_upload(ProgramBuilder& p,
                                             const std::shared_ptr<ov::op::v0::Constant>& op,
                                             const ov::Shape& const_shape,
                                             cldnn::data_types out_dtype,
                                             const cldnn::format& const_format,
                                             const cldnn::layout& const_layout) {
    PartialUploadDesc desc;

    const int64_t otd_ratio = p.get_config().get_offload_ratio();
    // Only routed expert weights are partially uploaded; shared experts stay fully resident.
    // ratio=0 (all resident) or ratio=100 (all on disk, invalid) → no partial upload.
    const bool partial_moe_const_upload = otd_ratio > 0 && otd_ratio < 100 && get_moe_constant_role(op) == MoEConstantRole::RoutedExpert;
    if (!partial_moe_const_upload || const_layout.bytes_count() == 0 || const_shape.empty() || const_shape[0] == 0) {
        return desc;
    }

    // otd_ratio is the % on disk; GPU-resident experts = total * (100 - ratio) / 100
    const size_t resident_expert_num = std::max<size_t>(1, const_shape[0] * static_cast<size_t>(100 - otd_ratio) / 100);

    desc.enabled = true;
    desc.upload_shape = const_shape;
    desc.upload_shape[0] = std::min<size_t>(const_shape[0], resident_expert_num);

    auto upload_layout = cldnn::layout(desc.upload_shape, out_dtype, const_format);
    auto upload_mem = p.get_engine().allocate_memory(upload_layout, false);
    // Reinterpret the smaller physical allocation as the full constant layout so the
    // graph sees the expected shape/layout. This is safe because:
    // 1. constant.cpp marks this data node with skip_device_transfer=true (partial_upload.enabled),
    //    so no host→device memcpy of the full size occurs.
    // 2. At runtime, OTD loads on-demand into the first `resident_expert_num` slots only.
    // 3. Model cache serialization uses weightless caching (bin_offset metadata) for these
    //    constants — it never reads the buffer contents via mem->buffer_ptr(). OTD requires
    //    weights_path to be set, which enables weightless caching for all data nodes.
    OPENVINO_ASSERT(upload_layout.bytes_count() <= const_layout.bytes_count(),
                    "Partial upload layout (", upload_layout.bytes_count(),
                    " bytes) exceeds full constant layout (", const_layout.bytes_count(), " bytes)");
    desc.memory = p.get_engine().reinterpret_buffer(*upload_mem, const_layout);
    desc.upload_bytes = upload_layout.bytes_count();

    get_partial_upload_log_state().log(op->get_friendly_name(),
                                       desc.upload_shape[0],
                                       const_shape[0],
                                       desc.upload_bytes,
                                       const_layout.bytes_count());
    return desc;
}

namespace {

// Returns total physical system RAM in bytes, or 0 if unavailable.
// Used only on integrated GPUs, where device "global memory" is shared with system RAM.
uint64_t get_total_system_ram_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        return static_cast<uint64_t>(status.ullTotalPhys);
    }
    return 0;
#elif defined(__linux__)
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) {
        return static_cast<uint64_t>(pages) * static_cast<uint64_t>(page_size);
    }
    return 0;
#else
    return 0;
#endif
}

// Recursively accumulates routed-expert weight bytes across the model, deduplicating by node identity.
void accumulate_moe_weight_bytes(const ov::Model& model,
                                 std::unordered_set<const ov::Node*>& visited,
                                 uint64_t& w_moe) {
    for (const auto& op : model.get_ops()) {
        auto constant = ov::as_type_ptr<ov::op::v0::Constant>(op);
        if (!constant || !visited.insert(constant.get()).second)
            continue;
        if (get_moe_constant_role(constant) == MoEConstantRole::RoutedExpert)
            w_moe += constant->get_byte_size();
    }
}

}  // namespace

size_t resolve_auto_offload_ratio(const ov::Model& model, const cldnn::device_info& info) {
    // Collect MoE routed-expert weight bytes (deduplicated by node identity).
    uint64_t w_moe = 0;
    std::unordered_set<const ov::Node*> visited;
    accumulate_moe_weight_bytes(model, visited, w_moe);

    // No offloadable MoE weights -> auto resolves to "no offload".
    if (w_moe == 0) {
        GPU_DEBUG_INFO << "[MOE OTD auto] no offloadable MoE routed-expert weights found; resolved offload_ratio=0" << std::endl;
        return 0;
    }

    // Memory budget: total device memory for dGPU; total system RAM for iGPU
    // (iGPU "global memory" is shared with RAM, so we use the smaller total RAM figure).
    uint64_t m_budget = info.max_global_mem_size;
    if (info.dev_type == cldnn::device_type::integrated_gpu) {
        const uint64_t total_ram = get_total_system_ram_bytes();
        if (total_ram > 0)
            m_budget = std::min<uint64_t>(m_budget, total_ram);
    }
    if (m_budget == 0) {
        GPU_DEBUG_INFO << "[MOE OTD auto] could not determine memory budget; resolved offload_ratio=0" << std::endl;
        return 0;
    }

    // Simple heuristic: allow MoE expert weights to consume at most 50% of device memory.
    constexpr double MOE_BUDGET_FRACTION = 0.5;
    const double budget_for_moe = static_cast<double>(m_budget) * MOE_BUDGET_FRACTION;

    size_t ratio;
    if (static_cast<double>(w_moe) <= budget_for_moe) {
        ratio = 0;  // everything fits, no offload needed
    } else {
        const double resident_fraction = budget_for_moe / static_cast<double>(w_moe);
        const long r = std::lround((1.0 - resident_fraction) * 100.0);
        // Cap AUTO ratio at AUTO_RATIO_MAX to avoid extreme offload that would hurt performance.
        constexpr size_t AUTO_RATIO_MAX = 70;
        ratio = static_cast<size_t>(std::clamp<long>(r, 0, static_cast<long>(AUTO_RATIO_MAX)));
    }

    GPU_DEBUG_INFO << "[MOE OTD auto] dev_type=" << (info.dev_type == cldnn::device_type::integrated_gpu ? "iGPU" : "dGPU")
                   << " m_budget=" << m_budget << " w_moe=" << w_moe
                   << " budget_for_moe=" << static_cast<uint64_t>(budget_for_moe)
                   << " -> resolved offload_ratio=" << ratio << std::endl;
    return ratio;
}

}  // namespace ov::intel_gpu
