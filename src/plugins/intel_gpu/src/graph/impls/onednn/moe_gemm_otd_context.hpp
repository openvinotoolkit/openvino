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

    size_t capacity = 0;  // Total expert count (= num_experts for 2GEMM)
    ov::op::internal::MOECompressed::Config moe_config{};
    std::vector<size_t> weight_bin_offsets;
    std::vector<bool> loaded_up;    // loaded_up[expert_id] = true if up weights loaded
    std::vector<bool> loaded_down;  // loaded_down[expert_id] = true if down weights loaded
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

    // Slot mapping from current iteration
    std::vector<size_t> current_slots;
    std::vector<int32_t> current_expert_ids;  // active expert IDs for this iteration

    Moe2GemmOtdContext(size_t lru_capacity,
                       const ov::op::internal::MOECompressed::Config& config,
                       std::vector<size_t> bin_offsets,
                       const std::filesystem::path& weights_path)
        : capacity(lru_capacity),
          moe_config(config),
          weight_bin_offsets(std::move(bin_offsets)),
          loaded_up(lru_capacity, false),
          loaded_down(lru_capacity, false),
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
                                                       const ov::op::internal::MOECompressed::Config& config,
                                                       const std::vector<size_t>& bin_offsets,
                                                       const std::filesystem::path& weights_path) {
        std::lock_guard<std::mutex> lock(_mutex);
        auto it = _contexts.find(moe_layer_id);
        if (it != _contexts.end()) {
            return it->second;
        }
        auto ctx = std::make_shared<Moe2GemmOtdContext>(capacity, config, bin_offsets, weights_path);
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
