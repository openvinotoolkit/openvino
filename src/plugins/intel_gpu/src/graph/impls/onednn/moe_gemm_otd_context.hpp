// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "intel_gpu/primitives/moe_otd_descriptor.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "ov_ops/moe_compressed.hpp"
#include "impls/ocl_v2/moe/moe_otd_runtime.hpp"

namespace cldnn {
namespace onednn {

/// @brief Shared OTD context for a pair of moe_gemm_up / moe_gemm_down primitives.
/// Both primitives within the same MOE layer share one context instance so that
/// expert weights loaded by gemm_up remain valid for gemm_down.
struct Moe2GemmOtdContext {
    // 2GEMM offset layout: [up_w, up_s, up_z, down_w, down_s, down_z]
    static constexpr size_t offset_count = 6;
    // Offset index constants
    static constexpr size_t UP_W = 0;
    static constexpr size_t UP_S = 1;
    static constexpr size_t UP_Z = 2;
    static constexpr size_t DOWN_W = 3;
    static constexpr size_t DOWN_S = 4;
    static constexpr size_t DOWN_Z = 5;

    size_t capacity = 0;      // LRU slot count (= lru_expert_num)
    size_t num_experts = 0;    // Total experts in the model (for per-expert-size computation)
    ov::op::internal::MOECompressed::Config moe_config{};
    std::vector<size_t> weight_bin_offsets;
    // Per-slot tracking: slot_contents_up[slot] = expert_id loaded there (-1 if empty)
    std::vector<int32_t> slot_contents_up;
    std::vector<int32_t> slot_contents_down;
    // Per-slot last-use tick for LRU eviction on the decode (non-relocating) path.
    std::vector<uint64_t> slot_used_up;
    std::vector<uint64_t> slot_used_down;
    uint64_t lru_tick = 0;
    std::unique_ptr<ov::intel_gpu::ocl::moe_otd::ParallelWeightReader> weight_reader;

    // Resident GPU memory buffers (one slot per LRU entry)
    struct ResidentBuffers {
        memory::ptr up_w;
        memory::ptr up_s;
        memory::ptr up_z;
        memory::ptr down_w;
        memory::ptr down_s;
        memory::ptr down_z;
    } resident{};
    bool bound = false;

    // One-slot scratch buffers used to swap expert weights between resident slots
    // (GPU->GPU) without a disk round-trip. Sized to the largest per-expert tensor of
    // each direction, allocated on first use and never reallocated afterwards so that
    // in-flight async copies keep a valid backing pointer.
    memory::ptr swap_tmp_up;
    memory::ptr swap_tmp_down;

    // Slot mapping from current iteration (set by gemm_up, reused by gemm_down)
    std::unordered_map<uint32_t, size_t> expert_to_slot;  // expert_id → LRU slot
    bool slots_valid = false;  // true after gemm_up loaded, cleared after gemm_down executes

    Moe2GemmOtdContext(size_t lru_capacity,
                       size_t total_experts,
                       const ov::op::internal::MOECompressed::Config& config,
                       std::vector<size_t> bin_offsets,
                       const std::filesystem::path& weights_path)
        : capacity(lru_capacity),
          num_experts(total_experts),
          moe_config(config),
          weight_bin_offsets(std::move(bin_offsets)),
          slot_contents_up(lru_capacity, -1),
          slot_contents_down(lru_capacity, -1),
          slot_used_up(lru_capacity, 0),
          slot_used_down(lru_capacity, 0),
          weight_reader(std::make_unique<ov::intel_gpu::ocl::moe_otd::ParallelWeightReader>(weights_path)) {}

    void bind(memory::ptr up_w, memory::ptr up_s, memory::ptr up_z,
              memory::ptr down_w, memory::ptr down_s, memory::ptr down_z) {
        resident.up_w = std::move(up_w);
        resident.up_s = std::move(up_s);
        resident.up_z = std::move(up_z);
        resident.down_w = std::move(down_w);
        resident.down_s = std::move(down_s);
        resident.down_z = std::move(down_z);
        bound = true;
    }

    // Load requested experts for the given direction (is_up=true → up weights, false → down weights).
    std::vector<size_t> load_experts(stream& s, const std::vector<uint32_t>& expert_ids, bool is_up);

    // Decode path: acquire a resident slot for each expert using a pure LRU cache WITHOUT
    // relocating experts to contiguous slots. Returns slot[i] for expert_ids[i]. Hits reuse
    // the existing slot (no copy); misses load from disk into a free/evicted slot. Callers
    // then issue one matmul per expert reading its weight from slot[i], avoiding the GPU->GPU
    // swaps required by the grouped (contiguous-slot) prefill path.
    std::vector<size_t> acquire_experts_lru(stream& s, const std::vector<uint32_t>& expert_ids, bool is_up);

    // Load a single expert's W/S/Z from disk into the given slot (read + transpose + upload).
    void load_expert_from_disk(stream& s, bool is_up, size_t slot, uint32_t expert);
};

/// @brief Global registry for Moe2GemmOtdContext instances, keyed by MOE layer base name.
/// Both gemm_up and gemm_down from the same MOE op share one context.
class Moe2GemmOtdRegistry {
public:
    static Moe2GemmOtdRegistry& instance() {
        static Moe2GemmOtdRegistry reg;
        return reg;
    }

    std::shared_ptr<Moe2GemmOtdContext> get_or_create(const std::string& moe_layer_id,
                                                       size_t capacity,
                                                       size_t num_experts,
                                                       const ov::op::internal::MOECompressed::Config& config,
                                                       const std::vector<size_t>& bin_offsets,
                                                       const std::filesystem::path& weights_path) {
        std::lock_guard<std::mutex> lock(_mutex);
        auto it = _contexts.find(moe_layer_id);
        if (it != _contexts.end()) {
            return it->second;
        }
        auto ctx = std::make_shared<Moe2GemmOtdContext>(capacity, num_experts, config, bin_offsets, weights_path);
        _contexts[moe_layer_id] = ctx;
        return ctx;
    }

    std::shared_ptr<Moe2GemmOtdContext> get(const std::string& moe_layer_id) {
        std::lock_guard<std::mutex> lock(_mutex);
        auto it = _contexts.find(moe_layer_id);
        if (it != _contexts.end()) {
            return it->second;
        }
        return nullptr;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(_mutex);
        _contexts.clear();
    }

private:
    Moe2GemmOtdRegistry() = default;
    std::mutex _mutex;
    std::unordered_map<std::string, std::shared_ptr<Moe2GemmOtdContext>> _contexts;
};

}  // namespace onednn
}  // namespace cldnn
