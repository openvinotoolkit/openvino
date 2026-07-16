// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_offload_constant.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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

    const size_t otd_ratio = p.get_config().get_offload_ratio();
    // Only routed expert weights are partially uploaded; shared experts stay fully resident.
    // ratio=0 (all resident) or ratio=100 (all on disk, invalid) → no partial upload.
    const bool partial_moe_const_upload = otd_ratio > 0 && otd_ratio < 100 && get_moe_constant_role(op) == MoEConstantRole::RoutedExpert;
    if (!partial_moe_const_upload || const_layout.bytes_count() == 0 || const_shape.empty() || const_shape[0] == 0) {
        return desc;
    }

    // otd_ratio is the % on disk; GPU-resident experts = total * (100 - ratio) / 100
    const size_t resident_expert_num = std::max<size_t>(1, const_shape[0] * (100 - otd_ratio) / 100);

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

// Returns the amount of currently-free physical system RAM in bytes, or 0 if unavailable.
// Used only on integrated GPUs, where device "global memory" is shared with system RAM.
uint64_t get_free_system_ram_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        return static_cast<uint64_t>(status.ullAvailPhys);
    }
    return 0;
#elif defined(__linux__)
    const long pages = sysconf(_SC_AVPHYS_PAGES);
    const long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) {
        return static_cast<uint64_t>(pages) * static_cast<uint64_t>(page_size);
    }
    return 0;
#else
    return 0;
#endif
}

// Recursively accumulates weight-constant bytes across the model and any subgraphs.
// w_total counts every Constant once (deduped by node identity); w_moe counts only
// routed-expert Constants (the offloadable subset).
void accumulate_weight_bytes(const ov::Model& model,
                             std::unordered_set<const ov::Node*>& visited,
                             uint64_t& w_total,
                             uint64_t& w_moe) {
    for (const auto& op : model.get_ops()) {
        if (auto sub = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(op)) {
            for (const auto& sub_model : sub->get_functions()) {
                accumulate_weight_bytes(*sub_model, visited, w_total, w_moe);
            }
        }
        auto constant = ov::as_type_ptr<ov::op::v0::Constant>(op);
        if (!constant)
            continue;
        if (!visited.insert(constant.get()).second)
            continue;  // already counted (shared constant referenced from multiple places)
        const uint64_t bytes = constant->get_byte_size();
        w_total += bytes;
        if (get_moe_constant_role(constant) == MoEConstantRole::RoutedExpert) {
            w_moe += bytes;
        }
    }
}

// Reads a scalar integer from a node input if it is a Constant, returning fallback otherwise.
int64_t read_scalar_input(const ov::Node& op, size_t input_idx, int64_t fallback) {
    if (input_idx >= op.get_input_size())
        return fallback;
    auto c = ov::as_type_ptr<ov::op::v0::Constant>(op.get_input_node_shared_ptr(input_idx));
    if (!c || ov::shape_size(c->get_shape()) < 1)
        return fallback;
    return c->cast_vector<int64_t>().front();
}

// Estimates the runtime memory reserve (KV cache + activations/scratch) from model structure.
// KV cache dominates for LLMs; it is derived from PagedAttention layers using their rt_info
// (k/v head size and kv-head count) and a per-layer effective context length. Layers that use
// sliding-window attention only retain a bounded window of KV, so their context is capped at the
// window size (PagedAttention input #10) rather than the full max_ctx -- this avoids grossly
// over-reserving for hybrid models (e.g. gemma-style 25 sliding + 5 full-attention layers, or
// qwen3.5-moe where only the full-attention layers keep a growing KV cache). Activations/scratch
// are approximated as a fixed fraction of the KV cache. This is model-derived (not a % of device
// memory) so it stays constant across machines of different memory sizes.
// Outputs the number of PagedAttention layers found and the raw KV bytes for diagnostics.
uint64_t estimate_runtime_reserve(const ov::Model& model, size_t max_ctx, size_t& pa_layers, uint64_t& kv_bytes_out) {
    uint64_t kv_bytes = 0;
    pa_layers = 0;
    // PagedAttentionExtension input layout: index 10 is the sliding_window scalar (0 = unlimited).
    constexpr size_t sliding_window_input_idx = 10;
    // Conservative KV element size: assume uncompressed f16 (2 bytes) to over-reserve rather
    // than risk OOM. If KV compression is active the real footprint is smaller, covered by slack.
    constexpr uint64_t kv_elem_bytes = 2;

    std::function<void(const ov::Model&)> walk = [&](const ov::Model& m) {
        for (const auto& op : m.get_ops()) {
            if (auto sub = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(op)) {
                for (const auto& sub_model : sub->get_functions()) {
                    walk(*sub_model);
                }
            }
            if (ov::is_type<ov::op::PagedAttentionExtension>(op)) {
                ++pa_layers;
                const auto& rt = op->get_rt_info();
                auto read = [&](const char* key) -> int64_t {
                    auto it = rt.find(key);
                    return it == rt.end() ? 0 : it->second.as<int64_t>();
                };
                const int64_t k_head_size = read("k_head_size");
                const int64_t v_head_size = read("v_head_size");
                const int64_t num_k_heads = read("num_k_heads");
                if (k_head_size > 0 && v_head_size > 0 && num_k_heads > 0) {
                    // Sliding-window layers only retain a bounded KV window; cap the effective
                    // context at the window size when present (>0). 0 means unlimited (full ctx).
                    uint64_t eff_ctx = max_ctx;
                    const int64_t window = read_scalar_input(*op, sliding_window_input_idx, 0);
                    if (window > 0) {
                        eff_ctx = std::min<uint64_t>(eff_ctx, static_cast<uint64_t>(window));
                    }
                    // Per layer: (K + V) * num_kv_heads * eff_ctx * bytes.
                    kv_bytes += static_cast<uint64_t>(k_head_size + v_head_size) * static_cast<uint64_t>(num_k_heads) *
                                eff_ctx * kv_elem_bytes;
                }
            }
        }
    };
    walk(model);

    kv_bytes_out = kv_bytes;
    // Add ~25% on top of KV for activations and scratch buffers.
    return kv_bytes + kv_bytes / 4;
}

}  // namespace

size_t resolve_auto_offload_ratio(const ov::Model& model, const cldnn::device_info& info) {
    uint64_t w_total = 0;
    uint64_t w_moe = 0;
    std::unordered_set<const ov::Node*> visited;
    accumulate_weight_bytes(model, visited, w_total, w_moe);

    // No offloadable MoE weights -> auto resolves to "no offload".
    if (w_moe == 0) {
        GPU_DEBUG_INFO << "[MOE OTD auto] no offloadable MoE routed-expert weights found; resolved offload_ratio=0" << std::endl;
        return 0;
    }

    // Memory budget: device memory for dGPU; for iGPU cap by free system RAM since the
    // device "global memory" is shared with (and overstated relative to) actual free RAM.
    uint64_t m_budget = info.max_global_mem_size;
    const bool is_igpu = info.dev_type == cldnn::device_type::integrated_gpu;
    if (is_igpu) {
        const uint64_t free_ram = get_free_system_ram_bytes();
        if (free_ram > 0) {
            m_budget = std::min<uint64_t>(m_budget, free_ram);
        }
    }
    if (m_budget == 0) {
        GPU_DEBUG_INFO << "[MOE OTD auto] could not determine memory budget; resolved offload_ratio=0" << std::endl;
        return 0;
    }

    const size_t max_ctx = 8192;
    const double safety = 0.85;
    const uint64_t w_fixed = w_total - w_moe;
    size_t pa_layers = 0;
    uint64_t kv_bytes = 0;
    const uint64_t reserve = estimate_runtime_reserve(model, max_ctx, pa_layers, kv_bytes);
    const uint64_t free_ram_now = get_free_system_ram_bytes();

    const double budget_for_moe =
        static_cast<double>(m_budget) * safety - static_cast<double>(w_fixed) - static_cast<double>(reserve);

    size_t ratio;
    if (budget_for_moe >= static_cast<double>(w_moe)) {
        ratio = 0;  // everything fits, no offload
    } else if (budget_for_moe <= 0.0) {
        ratio = 99;  // extreme pressure: keep a single LRU slot resident (100 == all-on-disk is invalid)
    } else {
        const double resident_fraction = budget_for_moe / static_cast<double>(w_moe);
        const long r = std::lround((1.0 - resident_fraction) * 100.0);
        ratio = static_cast<size_t>(std::clamp<long>(r, 0, 99));
    }

    constexpr double to_mib = 1.0 / (1024.0 * 1024.0);
    // Emit unconditionally (compile-time, once per model) so the decision is visible even in
    // release builds where GPU_DEBUG_INFO is compiled out.
    std::cout << "[MOE OTD auto] dev_type=" << (info.dev_type == cldnn::device_type::integrated_gpu ? "iGPU" : "dGPU")
              << " dev_mem=" << static_cast<uint64_t>(info.max_global_mem_size * to_mib) << "MiB"
              << " free_ram_now=" << static_cast<uint64_t>(free_ram_now * to_mib) << "MiB"
              << " M_budget=" << static_cast<uint64_t>(m_budget * to_mib) << "MiB"
              << " W_total=" << static_cast<uint64_t>(w_total * to_mib) << "MiB"
              << " W_moe=" << static_cast<uint64_t>(w_moe * to_mib) << "MiB"
              << " W_fixed=" << static_cast<uint64_t>(w_fixed * to_mib) << "MiB"
              << " Reserve=" << static_cast<uint64_t>(reserve * to_mib) << "MiB"
              << " (pa_layers=" << pa_layers << " kv=" << static_cast<uint64_t>(kv_bytes * to_mib) << "MiB)"
              << " budget_for_moe=" << static_cast<long long>(budget_for_moe * to_mib) << "MiB"
              << " SAFETY=" << safety << " max_ctx=" << max_ctx
              << " -> offload_ratio=" << ratio << std::endl;

    GPU_DEBUG_INFO << "[MOE OTD auto] M_budget=" << m_budget << " (dev_type=" << (info.dev_type == cldnn::device_type::integrated_gpu ? "iGPU" : "dGPU")
                   << "), W_total=" << w_total << ", W_moe=" << w_moe << ", W_fixed=" << w_fixed << ", Reserve=" << reserve
                   << ", SAFETY=" << safety << ", max_ctx=" << max_ctx << " -> resolved offload_ratio=" << ratio << std::endl;
    return ratio;
}

}  // namespace ov::intel_gpu
