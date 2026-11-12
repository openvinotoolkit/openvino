// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "expert_weight_provider.hpp"
#include "intel_gpu/runtime/les_context.hpp"
#include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"
#include "lru_cache.hpp"
#include "moe_otd_runtime.hpp"

namespace ov::intel_gpu::ocl::moe {

// Identity provider: every expert's weights are already resident in device memory
// for the network lifetime. This is the behaviour of the non-offloaded (ratio=0) path.
class ResidentExpertWeightProvider : public IExpertWeightProvider {
public:
    std::shared_ptr<IExpertWeightProvider> clone() const override { return std::make_shared<ResidentExpertWeightProvider>(); }
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
    void fill_routed_weight_views(cldnn::moe_weights& weights, RoutedWeightViews& views) override {
        views.weight[0] = weights.gate_w;
        views.scale[0] = weights.gate_s;
        views.zp[0] = weights.gate_z;
        views.weight[1] = weights.up_w;
        views.scale[1] = weights.up_s;
        views.zp[1] = weights.up_z;
        views.weight[2] = weights.down_w;
        views.scale[2] = weights.down_s;
        views.zp[2] = weights.down_z;
    }

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
    friend class ExpertPrefetcher;
public:
    OffloadExpertWeightProvider(size_t capacity,
                                const cldnn::MOECompressed::Config& config,
                                std::vector<size_t> weight_bin_offsets,
                                std::filesystem::path weights_path);
    ~OffloadExpertWeightProvider() override;

    size_t resident_capacity() const override {
        return _capacity;
    }
    bool is_offloaded() const override {
        return true;
    }

    void release() override;
    std::shared_ptr<IExpertWeightProvider> clone() const override;
    void set_les_context(cldnn::les_context* context, uint64_t epoch) override;
    void set_consumer(cldnn::event::ptr consumer) override;
    std::optional<ExpertDemand> begin_les_demand(const std::vector<uint32_t>& experts) override;
    void finish_les_demand(cldnn::stream& stream) override;
    void prefetch_les_demand(const std::vector<uint32_t>& experts, cldnn::stream& stream) override;
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

    moe_otd::ParallelWeightReader& weight_reader() {
        return _weight_reader;
    }

private:
    size_t _capacity = 0;
    cldnn::MOECompressed::Config _config{};
    std::vector<size_t> _weight_bin_offsets;
    std::filesystem::path _weights_path;
    std::shared_ptr<LRUCache> _cache;
    moe_otd::ParallelWeightReader _weight_reader;
    cldnn::moe_weights* _resident = nullptr;
    bool _bound = false;
    cldnn::les_context* _les_context = nullptr;
    uint64_t _les_epoch = 0;
    std::unique_ptr<ExpertSlotReservations> _reservations;
    std::optional<ExpertSlotReservations::Ticket> _reservation;
    std::unique_ptr<ExpertSlotReservations> _demand_shadow;
    std::optional<ExpertSlotReservations::Ticket> _shadow_reservation;
    bool _speculative_reservation = false;
    cldnn::event::ptr _consumer;
    std::shared_future<void> _upload_completion;
    std::shared_ptr<std::vector<cldnn::event::ptr>> _upload_events;
    bool _demand_finished = false;
    std::exception_ptr _prefetch_error;
    std::optional<cldnn::les_context::ticket> _prefetch_task;
    bool _prefetch_pending = false;
    uint64_t _prefetch_uploaded_bytes = 0;
    size_t _prefetch_expert_bytes = 0;
    void resolve_prefetch(const std::vector<uint32_t>& experts, bool schedule = true);
    std::optional<ExpertSlotLease> acquire_les(const std::vector<uint32_t>& experts, cldnn::stream& stream);
};

}  // namespace ov::intel_gpu::ocl::moe
