// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_offload_constant.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <unordered_set>

#include "openvino/op/paged_attention.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"

#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <dxgi1_4.h>
#    include <wrl/client.h>
#    pragma comment(lib, "dxgi.lib")
#endif

namespace ov::intel_gpu {

// Input indices 3..11 are routed expert weights/scales/zps (WEIGHT_0..ZP_2).
// Input indices 12..21 are shared expert weights (SHARED_GATE_WEIGHT..SHARED_GATE_GATE_WEIGHT).
// See MOE3GemmInputIndex in moe_3gemm_base.hpp for the authoritative enum.
static constexpr size_t ROUTED_INPUT_START = 3;
static constexpr size_t ROUTED_INPUT_END = 11;
static constexpr size_t SHARED_INPUT_START = 12;
static constexpr size_t SHARED_INPUT_END = 21;
static constexpr double AUTO_OFFLOAD_RATIO_FIT_SAFETY = 0.85;

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

#if defined(_WIN32)
bool is_luid_empty(const ov::device::LUID& luid) {
    return std::all_of(luid.luid.begin(), luid.luid.end(), [](uint8_t value) { return value == 0; });
}

bool match_luid(const ::LUID& dxgi_luid, const ov::device::LUID& ov_luid) {
    return std::memcmp(&dxgi_luid, ov_luid.luid.data(), sizeof(dxgi_luid)) == 0;
}

uint64_t query_dxgi_available_video_memory_bytes(const ov::device::LUID& luid) {
    if (is_luid_empty(luid))
        return 0;

    using Microsoft::WRL::ComPtr;

    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory))))
        return 0;

    ComPtr<IDXGIAdapter3> selected_adapter;
    for (UINT adapter_index = 0;; ++adapter_index) {
        ComPtr<IDXGIAdapter> adapter;
        if (factory->EnumAdapters(adapter_index, &adapter) == DXGI_ERROR_NOT_FOUND)
            break;

        DXGI_ADAPTER_DESC desc{};
        if (FAILED(adapter->GetDesc(&desc)))
            continue;

        if (!match_luid(desc.AdapterLuid, luid))
            continue;

        if (SUCCEEDED(adapter.As(&selected_adapter)))
            break;
    }

    if (!selected_adapter)
        return 0;

    uint64_t available_bytes = 0;
    for (UINT node_id = 0;; ++node_id) {
        DXGI_QUERY_VIDEO_MEMORY_INFO info{};
        if (selected_adapter->QueryVideoMemoryInfo(node_id, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info) != S_OK)
            break;

        if (info.Budget > info.CurrentUsage)
            available_bytes += static_cast<uint64_t>(info.Budget - info.CurrentUsage);
    }

    return available_bytes;
}
#else
uint64_t query_dxgi_available_video_memory_bytes(const ov::device::LUID&) {
    return 0;
}
#endif

uint64_t estimate_available_tracked_device_memory_bytes(const cldnn::engine& engine, uint64_t upper_bound) {
    uint64_t used_bytes = 0;
    for (const auto& stat : engine.get_memory_statistics()) {
        used_bytes += stat.second;
    }

    if (used_bytes >= upper_bound)
        return 0;

    return upper_bound - used_bytes;
}

struct MoEOffloadWeightStats {
    uint64_t total = 0;
    uint64_t routed = 0;
};

// Recursively accumulates weight-constant bytes across the model and any subgraphs.
// w_total counts every Constant once (deduped by node identity); w_moe counts only
// routed-expert Constants (the offloadable subset).
void accumulate_weight_bytes(const ov::Model& model,
                             std::unordered_set<const ov::Node*>& visited,
                             MoEOffloadWeightStats& stats) {
    for (const auto& op : model.get_ops()) {
        if (auto sub = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(op)) {
            for (const auto& sub_model : sub->get_functions()) {
                accumulate_weight_bytes(*sub_model, visited, stats);
            }
        }
        auto constant = ov::as_type_ptr<ov::op::v0::Constant>(op);
        if (!constant || !visited.insert(constant.get()).second)
            continue;
        stats.total += constant->get_byte_size();
        if (get_moe_constant_role(constant) == MoEConstantRole::RoutedExpert)
            stats.routed += constant->get_byte_size();
    }
}

MoEOffloadWeightStats collect_moe_offload_weight_stats(const ov::Model& model) {
    MoEOffloadWeightStats stats;
    std::unordered_set<const ov::Node*> visited;
    accumulate_weight_bytes(model, visited, stats);
    return stats;
}

size_t calculate_auto_offload_ratio(const MoEOffloadWeightStats& stats, uint64_t memory_budget) {
    if (stats.routed == 0 || memory_budget == 0)
        return 0;

    const uint64_t w_fixed = stats.total - stats.routed;

    const double budget_for_moe =
        static_cast<double>(memory_budget) * AUTO_OFFLOAD_RATIO_FIT_SAFETY - static_cast<double>(w_fixed);

    if (budget_for_moe >= static_cast<double>(stats.routed)) {
        return 0;  // everything fits, no offload needed
    }

    const double resident_fraction = budget_for_moe / static_cast<double>(stats.routed);
    return static_cast<size_t>(std::lround((1.0 - resident_fraction) * 100.0));
}

}  // namespace

size_t resolve_auto_offload_ratio_for_budget(const ov::Model& model, uint64_t memory_budget) {
    return calculate_auto_offload_ratio(collect_moe_offload_weight_stats(model), memory_budget);
}

size_t resolve_auto_offload_ratio(const ov::Model& model, cldnn::engine& engine) {
    const auto& info = engine.get_device_info();
    const auto stats = collect_moe_offload_weight_stats(model);

    // No offloadable MoE weights -> auto resolves to "no offload".
    if (stats.routed == 0) {
        GPU_DEBUG_INFO << "[MOE OTD auto] no offloadable MoE routed-expert weights found; resolved offload_ratio=0" << std::endl;
        return 0;
    }

    // Memory budget: device memory for dGPU. For iGPU, prefer OS budget if available and
    // otherwise use the GPU plugin's tracked allocations, as AUTO_BATCH does.
    uint64_t m_budget = info.max_global_mem_size;
    const bool is_igpu = info.dev_type == cldnn::device_type::integrated_gpu;
    std::string budget_source = "device_info";
    if (is_igpu) {
        const uint64_t dxgi_budget = query_dxgi_available_video_memory_bytes(info.luid);
        if (dxgi_budget > 0) {
            m_budget = std::min<uint64_t>(m_budget, dxgi_budget);
            budget_source = "dxgi_budget";
        } else {
            const uint64_t tracked_budget = estimate_available_tracked_device_memory_bytes(engine, m_budget);
            if (tracked_budget > 0) {
                m_budget = tracked_budget;
                budget_source = "tracked_mem_stats";
            }
        }
    }
    if (m_budget == 0) {
        GPU_DEBUG_INFO << "[MOE OTD auto] could not determine memory budget; resolved offload_ratio=0" << std::endl;
        return 0;
    }

    const uint64_t w_fixed = stats.total - stats.routed;
    const double budget_for_moe = static_cast<double>(m_budget) * AUTO_OFFLOAD_RATIO_FIT_SAFETY - static_cast<double>(w_fixed);
    const size_t ratio = calculate_auto_offload_ratio(stats, m_budget);

    std::cout << "[MOE OTD auto] dev_type=" << (info.dev_type == cldnn::device_type::integrated_gpu ? "iGPU" : "dGPU")
                   << " m_budget=" << m_budget
                   << " budget_source=" << budget_source
                   << " w_total=" << stats.total
                   << " w_moe=" << stats.routed
                   << " w_fixed=" << w_fixed
                   << " budget_for_moe=" << static_cast<long long>(budget_for_moe)
                   << " -> resolved offload_ratio=" << ratio << std::endl;
    return ratio;
}

}  // namespace ov::intel_gpu
