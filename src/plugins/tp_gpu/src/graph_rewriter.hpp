// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/model.hpp"
#include "tp_gpu/tp_config.hpp"

namespace ov {
namespace tp_gpu {

class TPDeviceCoordinator;

/// \brief Sharding plan produced by model analysis.
struct ShardingPlan {
    struct LinearDesc {
        std::string matmul_name;
        int layer_idx;
        enum Role { Q_PROJ, K_PROJ, V_PROJ, O_PROJ, GATE_PROJ, UP_PROJ, DOWN_PROJ } role;
        /// Column-parallel splits the weight's output dim, row-parallel its
        /// input dim -- and only row-parallel needs an AllReduce afterwards.
        bool is_column_parallel;
        /// Column-parallel biases are sharded with the weight; row-parallel
        /// ones stay whole, because their Add runs after the AllReduce.
        bool has_bias = false;
    };

    std::vector<LinearDesc> linears;

    /// Which attention op anchors the layers. The two differ in how heads
    /// reach the attention: SDPA broadcasts KV heads up to the query head
    /// count in the graph, PagedAttention takes flattened
    /// [tokens, heads * head_dim] operands and does the grouping itself.
    enum class AttentionBackend { NONE, SDPA, PA } attention_backend = AttentionBackend::NONE;

    int num_layers = 0;
    int num_heads = 0;          ///< Q attention heads
    int num_kv_heads = 0;       ///< KV attention heads (for GQA)
    int head_dim = 0;
    int hidden_size = 0;
    int intermediate_size = 0;

    /// The vocabulary projection, when it can be split across ranks.
    /// Empty when the model has no such projection, when its vocabulary does
    /// not divide by the world size (the gather assumes equal slices), or when
    /// its shape is not static. Every rank then keeps the whole projection.
    std::string lm_head_name;
    int64_t lm_head_vocab = 0;
};

/// \brief Analyzes a transformer model and rewrites it for tensor parallelism.
class GraphRewriter {
public:
    /// Analyze the model and produce a sharding plan.
    static ShardingPlan analyze(const std::shared_ptr<const ov::Model>& model);

    /// Clone the model and shard it for `rank`.
    static std::shared_ptr<ov::Model> rewrite(const std::shared_ptr<const ov::Model>& model,
                                              const ShardingPlan& plan,
                                              uint32_t rank,
                                              uint32_t tp_degree,
                                              const TPConfig& config = TPConfig{});

    /// How many collectives `rewrite` will insert.
    static int count_collectives(const ShardingPlan& plan, int tp_degree, const TPConfig& config = TPConfig{});

    /// Whether the vocabulary projection is split across `tp_degree` ranks.
    static bool shards_lm_head(const ShardingPlan& plan, int tp_degree, const TPConfig& config = TPConfig{});

    /// Ids of the variables `rewrite` shards along the kv-head axis.
    /// Their per-rank states each hold a slice of the KV cache, so a caller
    /// that reads or writes whole state tensors has to see them gathered and
    /// scattered. Every other variable is replicated on all ranks.
    static std::vector<std::string> sharded_state_ids(const std::shared_ptr<const ov::Model>& model,
                                                      const ShardingPlan& plan);
};

}  // namespace tp_gpu
}  // namespace ov
