// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_gemm_otd_context.hpp"

#include <chrono>
#include <cstring>

#include "impls/ocl_v2/moe/moe_otd_runtime.hpp"

namespace cldnn {
namespace onednn {

// Transpose scale/zp from [OC, G] to [G, OC] layout for oneDNN consumption.
// This is the 2GEMM variant that computes OC from the actual per-expert memory size.
static void transpose_scale_or_zp(const layout& mem_layout,
                                  std::vector<uint8_t>& payload,
                                  size_t per_expert_size,
                                  size_t oc,
                                  size_t group_count) {
    const auto elem_type = ov::element::Type(mem_layout.data_type);
    const size_t elem_size = elem_type.bitwidth() / 8;
    if (elem_size == 0 || oc == 0 || group_count == 0)
        return;

    const size_t expected_size = oc * group_count * elem_size;
    if (expected_size != per_expert_size)
        return;  // Dimensions don't match, skip transpose

    std::vector<uint8_t> transposed(per_expert_size, 0);
    for (size_t o = 0; o < oc; o++) {
        for (size_t g = 0; g < group_count; g++) {
            const size_t src_elem_idx = o * group_count + g;
            const size_t dst_elem_idx = g * oc + o;
            std::memcpy(transposed.data() + dst_elem_idx * elem_size,
                        payload.data() + src_elem_idx * elem_size,
                        elem_size);
        }
    }
    payload.swap(transposed);
}

std::vector<size_t> Moe2GemmOtdContext::load_experts(stream& s, const std::vector<uint32_t>& expert_ids, bool is_up) {
    using namespace ov::intel_gpu::ocl;

    std::vector<size_t> slots(expert_ids.size());
    auto* perf = moe_otd::get_perf_counters();
    auto& loaded_set = is_up ? loaded_up : loaded_down;

    // Select tensor info for this direction
    struct TensorInfo {
        size_t offset_idx;
        memory::ptr mem;
        bool needs_transpose;
    };
    std::array<TensorInfo, 3> tensors = is_up
        ? std::array<TensorInfo, 3>{{
            {UP_W, resident.up_w, false},
            {UP_S, resident.up_s, true},
            {UP_Z, resident.up_z, true},
          }}
        : std::array<TensorInfo, 3>{{
            {DOWN_W, resident.down_w, false},
            {DOWN_S, resident.down_s, true},
            {DOWN_Z, resident.down_z, true},
          }};

    for (size_t i = 0; i < expert_ids.size() && i < capacity; i++) {
        const uint32_t expert = expert_ids[i];
        const size_t slot = expert;  // Direct mapping: expert_id = position in weight buffer
        slots[i] = slot;

        if (loaded_set[expert]) {
            if (perf)
                perf->gpu_hits.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        if (perf)
            perf->gpu_misses.fetch_add(1, std::memory_order_relaxed);

        for (auto& [offset_idx, mem, needs_transpose] : tensors) {
            if (!mem)
                continue;

            const auto total_bytes = mem->get_layout().bytes_count();
            const size_t per_expert_size = total_bytes / capacity;
            const size_t base_offset = weight_bin_offsets[offset_idx];
            const size_t src_offset = base_offset + expert * per_expert_size;
            const size_t dst_offset = slot * per_expert_size;

            std::vector<uint8_t> payload(per_expert_size);

            if (perf) {
                auto t0 = std::chrono::steady_clock::now();
                weight_reader->read(reinterpret_cast<char*>(payload.data()), per_expert_size, src_offset);
                auto t1 = std::chrono::steady_clock::now();
                perf->disk_io_ns.fetch_add(
                    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()),
                    std::memory_order_relaxed);
            } else {
                weight_reader->read(reinterpret_cast<char*>(payload.data()), per_expert_size, src_offset);
            }

            // Transpose scale/zp from [OC, G] to [G, OC]
            // Derive OC and G from the actual memory shape [E, OC, G] rather than config
            if (needs_transpose) {
                const auto& shape = mem->get_layout().get_shape();
                const size_t tensor_oc = (shape.size() >= 2) ? shape[1] : 0;
                const size_t tensor_gc = (shape.size() >= 3) ? shape[2] : 0;
                if (perf) {
                    auto t0 = std::chrono::steady_clock::now();
                    transpose_scale_or_zp(mem->get_layout(), payload, per_expert_size, tensor_oc, tensor_gc);
                    auto t1 = std::chrono::steady_clock::now();
                    perf->transpose_ns.fetch_add(
                        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()),
                        std::memory_order_relaxed);
                } else {
                    transpose_scale_or_zp(mem->get_layout(), payload, per_expert_size, tensor_oc, tensor_gc);
                }
            }

            // Copy to GPU
            if (perf) {
                auto t0 = std::chrono::steady_clock::now();
                mem->copy_from(s, payload.data(), 0, dst_offset, per_expert_size, true);
                auto t1 = std::chrono::steady_clock::now();
                perf->gpu_copy_ns.fetch_add(
                    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()),
                    std::memory_order_relaxed);
                perf->tensor_load_count.fetch_add(1, std::memory_order_relaxed);
            } else {
                mem->copy_from(s, payload.data(), 0, dst_offset, per_expert_size, true);
            }
        }

        loaded_set[expert] = true;
    }

    return slots;
}

}  // namespace onednn
}  // namespace cldnn
