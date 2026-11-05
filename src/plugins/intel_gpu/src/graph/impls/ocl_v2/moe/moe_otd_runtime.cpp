// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_otd_runtime.hpp"

#include <array>
#include <chrono>
#include <cstdlib>
#include <utility>

#include "intel_gpu/runtime/debug_configuration.hpp"

namespace ov::intel_gpu::ocl::moe_otd {

void OtdPerfCounters::dump() const {
    const auto hits = gpu_hits.load(std::memory_order_relaxed);
    const auto misses = gpu_misses.load(std::memory_order_relaxed);
    const auto total = hits + misses;
    const auto loads = tensor_load_count.load(std::memory_order_relaxed);
    GPU_DEBUG_INFO << "[OTD_PERF] gpu_hits=" << hits << ", gpu_misses=" << misses << ", gpu_hit_rate=" << (total > 0 ? 100.0 * hits / total : 0.0) << "%"
                   << ", tensor_loads=" << loads << ", avg_disk_io_us=" << (loads > 0 ? disk_io_ns.load(std::memory_order_relaxed) / 1000 / loads : 0)
                   << ", avg_gpu_copy_us=" << (loads > 0 ? gpu_copy_ns.load(std::memory_order_relaxed) / 1000 / loads : 0)
                   << ", total_disk_io_ms=" << disk_io_ns.load(std::memory_order_relaxed) / 1000000
                   << ", total_gpu_copy_ms=" << gpu_copy_ns.load(std::memory_order_relaxed) / 1000000
                   << ", batched_fallbacks=" << batched_fallbacks.load(std::memory_order_relaxed)
                   << ", grouped_fallbacks=" << grouped_fallbacks.load(std::memory_order_relaxed)
                   << ", created_onednn_kernels=" << created_onednn_kernels.load(std::memory_order_relaxed) << std::endl;
}

OtdPerfCounters* get_perf_counters() {
    static bool enabled = std::getenv("MOE_OTD_PERF_LOG") != nullptr;
    if (!enabled) {
        return nullptr;
    }

    static OtdPerfCounters counters;
    static bool registered = [] {
        std::atexit([] {
            counters.dump();
        });
        return true;
    }();
    (void)registered;
    return &counters;
}

void fill_weights_memory(cldnn::stream& exec_stream,
                         const cldnn::MOECompressed::Config& config,
                         const std::vector<size_t>& weight_bin_offsets,
                         ParallelWeightReader& weight_reader,
                         cldnn::moe_weights& slots,
                         const cldnn::moe_weights& sources,
                         const std::vector<uint32_t>& experts_list,
                         const std::vector<size_t>& lru_experts) {
    const auto num_expert = static_cast<size_t>(config.num_expert);
    OPENVINO_ASSERT(num_expert > 0, "Invalid expert count");
    OPENVINO_ASSERT(weight_bin_offsets.size() == cldnn::moe_3gemm_fused_compressed::serialized_weight_offset_count, "Unexpected number of MOE weight offsets");
    OPENVINO_ASSERT(experts_list.size() == lru_experts.size(), "Each loaded expert needs a resident slot");

    static const std::array<const char*, cldnn::moe_3gemm_fused_compressed::serialized_weight_offset_count> weight_names = {{"gate_w", "up_w", "down_w"}};
    const std::array<cldnn::memory_ptr, cldnn::moe_3gemm_fused_compressed::serialized_weight_offset_count> weight_slots = {{slots.gate_w, slots.up_w, slots.down_w}};
    // {resident tensor with every expert, slot buffer}; the source is null for an absent or shared by all the experts tensor
    const std::array<std::pair<cldnn::memory_ptr, cldnn::memory_ptr>, 6> param_slots = {{{sources.gate_s, slots.gate_s},
                                                                                         {sources.up_s, slots.up_s},
                                                                                         {sources.down_s, slots.down_s},
                                                                                         {sources.gate_z, slots.gate_z},
                                                                                         {sources.up_z, slots.up_z},
                                                                                         {sources.down_z, slots.down_z}}};

    auto* perf = get_perf_counters();
    const size_t weight_file_size = weight_reader.file_size();
    std::vector<uint8_t> payload;

    for (size_t i = 0; i < experts_list.size(); i++) {
        const size_t expert = experts_list[i];
        const size_t slot = lru_experts[i];

        // The weights are consumed exactly as stored in the weights file (checked at compile time)
        for (size_t w = 0; w < weight_slots.size(); w++) {
            const auto& mem = weight_slots[w];
            if (!mem) {
                continue;
            }
            const size_t per_expert_size = mem->get_layout().bytes_count() / num_expert;
            const size_t src_offset = weight_bin_offsets[w] + expert * per_expert_size;
            OPENVINO_ASSERT(src_offset <= weight_file_size && per_expert_size <= weight_file_size - src_offset,
                            "Read range out of file for tensor ",
                            weight_names[w],
                            ": src_offset=",
                            src_offset,
                            ", per_expert_size=",
                            per_expert_size,
                            ", file_size=",
                            weight_file_size,
                            ", expert=",
                            expert);
            payload.resize(per_expert_size);
            const auto t0 = std::chrono::steady_clock::now();
            weight_reader.read(reinterpret_cast<char*>(payload.data()), per_expert_size, src_offset);
            const auto t1 = std::chrono::steady_clock::now();
            mem->copy_from(exec_stream, payload.data(), 0, slot * per_expert_size, per_expert_size, true);
            if (perf) {
                const auto t2 = std::chrono::steady_clock::now();
                perf->disk_io_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()),
                                           std::memory_order_relaxed);
                perf->gpu_copy_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()),
                                            std::memory_order_relaxed);
                perf->tensor_load_count.fetch_add(1, std::memory_order_relaxed);
            }
        }

        // Scales and zero points are already in their final form (converted and reordered by the regular pipeline)
        for (const auto& [source, slot_mem] : param_slots) {
            if (!source) {
                continue;
            }
            const size_t per_expert_size = source->get_layout().bytes_count() / num_expert;
            slot_mem->copy_from(exec_stream, *source, expert * per_expert_size, slot * per_expert_size, per_expert_size, true);
        }
    }
}

}  // namespace ov::intel_gpu::ocl::moe_otd
