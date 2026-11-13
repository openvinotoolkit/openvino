// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <memory>
#include <vector>

#include "intel_gpu/runtime/event.hpp"
#include "intel_gpu/runtime/execution_config.hpp"
#include "intel_gpu/runtime/memory.hpp"

namespace cldnn {
class engine;
}

namespace ov::intel_gpu::ocl::moe {

class OffloadExpertWeightProvider;

struct PrefetchTask {
    uint32_t target_layer_idx = 0;
    uint64_t generation = 0;
    cldnn::memory::ptr prediction_event_mem;
    cldnn::event::ptr prediction_ready_event;
    std::weak_ptr<OffloadExpertWeightProvider> target_provider;
    std::shared_ptr<std::atomic<bool>> consumed;
};

class ExpertPrefetcher {
public:
    ExpertPrefetcher(cldnn::engine& engine, const cldnn::ExecutionConfig& config);
    ~ExpertPrefetcher();
    ExpertPrefetcher(const ExpertPrefetcher&) = delete;
    ExpertPrefetcher& operator=(const ExpertPrefetcher&) = delete;

    void bind(const std::shared_ptr<OffloadExpertWeightProvider>& provider);
    cldnn::event::ptr make_ready_event();
    void submit_prefetch(PrefetchTask task);
    cldnn::event::ptr demand_async(OffloadExpertWeightProvider& provider, const std::vector<uint32_t>& missing_ids);
    void promote(OffloadExpertWeightProvider& provider, const std::vector<uint32_t>& topk_ids);
    void advance(uint64_t generation, uint32_t layer_idx);
    void rethrow_if_failed() const;
    void drain();
    void cancel_all();

private:
    struct State;
    std::unique_ptr<State> m_state;
};

}