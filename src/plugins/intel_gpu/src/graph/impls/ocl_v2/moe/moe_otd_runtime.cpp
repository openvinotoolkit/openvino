// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_otd_runtime.hpp"

#include <array>
#include <chrono>
#include <cstdlib>

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

void fill_weights_memory(cldnn::stream& exec_stream, ParallelWeightReader& weight_reader, const std::vector<SlicedTensor>& tensors, size_t index, size_t slot) {
    auto* perf = get_perf_counters();
    // Sources and slots outlive the device copies
    const bool blocking = exec_stream.get_queue_type() != cldnn::QueueTypes::in_order;
    std::vector<uint8_t> payload;

    for (const auto& tensor : tensors) {
        if (tensor.source) {
            tensor.slots->copy_from(exec_stream, *tensor.source, index * tensor.slice_size, slot * tensor.slice_size, tensor.slice_size, blocking);
            continue;
        }
        const size_t weight_file_size = weight_reader.file_size();
        const size_t src_offset = tensor.file_offset + (index * tensor.slice_size);
        OPENVINO_ASSERT(src_offset <= weight_file_size && tensor.slice_size <= weight_file_size - src_offset,
                        "Read range out of file: src_offset=",
                        src_offset,
                        ", slice_size=",
                        tensor.slice_size,
                        ", file_size=",
                        weight_file_size,
                        ", index=",
                        index);
        payload.resize(tensor.slice_size);
        const auto t0 = std::chrono::steady_clock::now();
        weight_reader.read(reinterpret_cast<char*>(payload.data()), tensor.slice_size, src_offset);
        const auto t1 = std::chrono::steady_clock::now();
        tensor.slots->copy_from(exec_stream, payload.data(), 0, slot * tensor.slice_size, tensor.slice_size, true);
        if (perf) {
            const auto t2 = std::chrono::steady_clock::now();
            perf->disk_io_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()), std::memory_order_relaxed);
            perf->gpu_copy_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()),
                                        std::memory_order_relaxed);
            perf->tensor_load_count.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

}  // namespace ov::intel_gpu::ocl::moe_otd
