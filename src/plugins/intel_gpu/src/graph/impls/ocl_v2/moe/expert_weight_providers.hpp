// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "expert_weight_provider.hpp"
#include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"
#include "lru_cache.hpp"
#include "moe_otd_runtime.hpp"

namespace ov::intel_gpu::ocl::moe {

// Identity provider: every expert's weights are already resident in device memory
// for the network lifetime. This is the behaviour of the non-offloaded (ratio=0) path.
class ResidentExpertWeightProvider : public IExpertWeightProvider {
public:
    void release() override {}
    size_t resident_capacity() const override {
        return 0;
    }
    bool is_offloaded() const override {
        return false;
    }
    void bind(cldnn::moe_weights& /*resident*/) override {}
    bool is_bound() const override {
        return true;
    }

    std::optional<ExpertSlotLease> try_acquire_simultaneous(const std::vector<uint32_t>& experts,
                                                            const RoutedWeightViews& inputs,
                                                            const RoutedWeightViews& slots,
                                                            cldnn::stream& stream) override;
    size_t acquire_one(uint32_t expert, const RoutedWeightViews& inputs, const RoutedWeightViews& slots, cldnn::stream& stream) override;
};

// Offload (OTD) provider: expert weights live on disk and are streamed on demand
// into a bounded set of device-resident slots managed by an LRU cache.
// The provider deduplicates the requested experts, assigns/streams slots, and
// returns the slot index per requested expert.
//
// The resident weight buffers are not owned here; they are bound at runtime via
// bind() because in this back-end they alias the primitive's
// input memory. The scales and zero points stay fully resident in the primitive inputs; the per-expert ones
// are copied on a miss into slot buffers that the primitive instance owns and passes to every acquisition.
// Loading metadata (config / bin offsets / weights path) is captured
// at construction from graph-build time information.
class OffloadExpertWeightProvider : public IExpertWeightProvider {
public:
    OffloadExpertWeightProvider(size_t capacity,
                                const cldnn::MOECompressed::Config& config,
                                std::vector<size_t> weight_bin_offsets,
                                std::filesystem::path weights_path);

    size_t resident_capacity() const override {
        return _capacity;
    }
    bool is_offloaded() const override {
        return true;
    }

    void release() override {}
    void bind(cldnn::moe_weights& resident) override;
    bool is_bound() const override {
        return _bound;
    }
    std::optional<ExpertSlotLease> try_acquire_simultaneous(const std::vector<uint32_t>& experts,
                                                            const RoutedWeightViews& inputs,
                                                            const RoutedWeightViews& slot_buffers,
                                                            cldnn::stream& stream) override;
    size_t acquire_one(uint32_t expert, const RoutedWeightViews& inputs, const RoutedWeightViews& slots, cldnn::stream& stream) override;

    LRUCache& cache() {
        return *_cache;
    }

    moe_otd::ParallelWeightReader& weight_reader() {
        return _weight_reader;
    }

private:
    // The engine may re-allocate an internal buffer, and then the experts loaded into it are gone
    void reset_cache_if_slots_changed(const RoutedWeightViews& slots);
    void load(uint32_t expert, size_t slot, const RoutedWeightViews& inputs, const RoutedWeightViews& slots, cldnn::stream& stream);

    size_t _capacity = 0;
    cldnn::MOECompressed::Config _config{};
    std::vector<size_t> _weight_bin_offsets;
    std::filesystem::path _weights_path;
    std::shared_ptr<LRUCache> _cache;
    moe_otd::ParallelWeightReader _weight_reader;
    bool _bound = false;
    // Handles of the slot buffers the LRU describes, compared only to detect a re-allocation
    std::vector<const void*> _slot_handles;
};

}  // namespace ov::intel_gpu::ocl::moe
