// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "expert_weight_provider.hpp"
#include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"
#include "lru_cache.hpp"

namespace ov::intel_gpu::ocl::moe {

// Identity provider: every expert's weights are already resident in device memory
// for the network lifetime. This is the behaviour of the non-offloaded (ratio=0) path.
class ResidentExpertWeightProvider : public IExpertWeightProvider {
public:
    size_t resident_capacity() const override {
        return 0;
    }
    bool is_offloaded() const override {
        return false;
    }

    // Lease API: resident provider is identity mapping
    std::optional<ExpertSlotLease> try_acquire_simultaneous(const std::vector<uint32_t>& experts, cldnn::stream& stream) override;
    size_t acquire_one(uint32_t expert, cldnn::stream& stream) override;
};

// Offload (OTD) provider: expert weights live on disk and are streamed on demand
// into a bounded set of device-resident slots managed by an LRU cache.
// The provider deduplicates the requested experts, assigns/streams slots, and
// returns the slot index per requested expert.
//
// The resident device buffers are not owned here; they are bound at runtime via
// bind() because in this back-end they alias the primitive's
// input memory. Loading metadata (config / bin offsets / weights path) is captured
// at construction from graph-build time information.
class OffloadExpertWeightProvider : public IExpertWeightProvider {
public:
    OffloadExpertWeightProvider(size_t capacity, const cldnn::MOECompressed::Config& config, std::vector<size_t> weight_bin_offsets, std::string weights_path);

    size_t resident_capacity() const override {
        return _capacity;
    }
    bool is_offloaded() const override {
        return true;
    }

    // Lease API
    void bind(cldnn::moe_weights& resident) override;
    bool is_bound() const override {
        return _bound;
    }
    void fill_routed_weight_views(cldnn::moe_weights& weights, RoutedWeightViews& views) override;
    std::optional<ExpertSlotLease> try_acquire_simultaneous(const std::vector<uint32_t>& experts, cldnn::stream& stream) override;
    size_t acquire_one(uint32_t expert, cldnn::stream& stream) override;

    LRUCache& cache() {
        return *_cache;
    }

private:
    size_t _capacity = 0;
    cldnn::MOECompressed::Config _config{};
    std::vector<size_t> _weight_bin_offsets;
    std::string _weights_path;
    std::shared_ptr<LRUCache> _cache;
    cldnn::moe_weights* _resident = nullptr;
    bool _bound = false;
};

}  // namespace ov::intel_gpu::ocl::moe
