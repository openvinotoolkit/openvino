// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/model.hpp"

namespace ov {
namespace tp_gpu {

class TPCoordination;

/// \brief Sharding plan produced by model analysis.
///
/// Describes which linear layers to shard and their parallelism strategy.
struct ShardingPlan {
    struct LinearDesc {
        std::string matmul_name;
        int layer_idx;
        enum Role { Q_PROJ, K_PROJ, V_PROJ, O_PROJ, GATE_PROJ, UP_PROJ, DOWN_PROJ } role;
        /// Column-parallel: shard output dim (weight axis 0).
        /// Row-parallel:    shard input dim  (weight axis 1).
        bool is_column_parallel;
        /// Whether the projection carries a bias.  Column-parallel biases are
        /// sharded along with the weight; row-parallel ones stay whole and are
        /// applied after the AllReduce.
        bool has_bias = false;
    };

    std::vector<LinearDesc> linears;

    int num_layers = 0;
    int num_heads = 0;          ///< Q attention heads
    int num_kv_heads = 0;       ///< KV attention heads (for GQA)
    int head_dim = 0;
    int hidden_size = 0;
    int intermediate_size = 0;
};

/// \brief Analyzes a transformer model and rewrites it for tensor parallelism.
///
/// Supports LLaMA-family models with the naming convention:
///   layers.{N}.self_attn.{q,k,v,o}_proj
///   layers.{N}.mlp.{gate,up,down}_proj
///
/// Weight sharding strategy (with transpose_b=true, weight shape is [out, in]):
///   - Column-parallel (q/k/v/gate/up_proj): slice weight axis 0 (output dim)
///   - Row-parallel (o/down_proj): slice weight axis 1 (input dim)
///
/// Additional modifications per rank:
///   - Reshape constants after q/k/v_proj: head count adjusted
///   - KV cache Variable shapes: kv_heads dimension adjusted
class GraphRewriter {
public:
    /// Analyze the model and produce a sharding plan.
    static ShardingPlan analyze(const std::shared_ptr<const ov::Model>& model);

    /// Clone the model and apply weight sharding for the given rank.
    /// Returns a new model with sharded weights and adjusted shapes.
    static std::shared_ptr<ov::Model> rewrite(const std::shared_ptr<const ov::Model>& model,
                                              const ShardingPlan& plan,
                                              uint32_t rank,
                                              uint32_t tp_degree,
                                              const std::shared_ptr<TPCoordination>& coordination);

    /// Count how many AllReduce collectives will be created (= number of row-parallel linears).
    static int count_collectives(const ShardingPlan& plan);
};

}  // namespace tp_gpu
}  // namespace ov
