// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mutex>
#include <atomic>
#include <unordered_map>

#include "lookahead_router.hpp"
#include "moe_lookahead_prefetcher.hpp"
#include "../expert_weight_providers.hpp"

namespace ov::intel_gpu::ocl::moe {

struct LookaheadLayerDesc {
    const cldnn::primitive_inst* moe_inst = nullptr;
    const cldnn::primitive_inst* router_fc_inst = nullptr;
    cldnn::memory::ptr weight_mem;
    uint32_t num_experts = 0;
    uint32_t hidden_size = 0;
    bool is_supported = false;
};

class LookaheadContext {
public:
    struct Prediction {
        cldnn::memory::ptr hidden;
        cldnn::event::ptr hidden_ready;
        cldnn::memory::ptr predicted_indices;
        cldnn::memory::ptr predicted_weights;
        cldnn::event::ptr prediction_ready;
        std::shared_ptr<std::atomic<bool>> consumed;
        bool submitted = false;
    };

    explicit LookaheadContext(cldnn::network& network);
    ~LookaheadContext();
    void build_once(cldnn::network& network);
    const LookaheadLayerDesc* get_next_layer(const cldnn::primitive_inst* current_moe) const;
    ExpertPrefetcher& prefetcher() { return *m_prefetcher; }
    uint64_t epoch() const { return m_epoch; }
    const std::vector<cldnn::program::les_router_link>& links() const { return m_links; }

    void enter(const cldnn::primitive_id& id);
    void demand_partitioned(const cldnn::primitive_id& id);
    void pause_for_prefill();
    std::shared_ptr<IExpertWeightProvider> bind(const cldnn::primitive_id& id,
                                               std::shared_ptr<IExpertWeightProvider> provider);

    Prediction prediction(const cldnn::primitive_id& id) const {
        auto found = m_predictions.find(id);
        return found == m_predictions.end() ? Prediction{} : found->second;
    }

    void predict(const LookaheadLayerDesc& next_layer,
                 const cldnn::memory::ptr& hidden,
                 const cldnn::event::ptr& ready);
    void prefetch(const cldnn::primitive_inst* current_moe);

private:
    cldnn::network& m_network;
    cldnn::stream::ptr m_compute_stream;
    std::vector<cldnn::program::les_router_link> m_links;
    std::once_flag m_init_flag;
    std::unordered_map<cldnn::primitive_id, uint32_t> m_layer_indices;
    std::unordered_map<cldnn::primitive_id, size_t> m_source_links;
    std::vector<LookaheadLayerDesc> m_layers;
    std::unordered_map<const cldnn::primitive_inst*, size_t> m_moe_to_next_index;
    uint64_t m_epoch = 0;
    std::optional<uint32_t> m_cursor;
    std::unordered_map<cldnn::primitive_id, std::shared_ptr<OffloadExpertWeightProvider>> m_providers;
    std::unordered_map<cldnn::primitive_id, std::unique_ptr<LookaheadRouter>> m_routers;
    std::unordered_map<cldnn::primitive_id, Prediction> m_predictions;
    std::unique_ptr<ExpertPrefetcher> m_prefetcher;
};

class LookaheadRegistry {
public:
    static std::shared_ptr<LookaheadContext> acquire(cldnn::network& network);
};

}