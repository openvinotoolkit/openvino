// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>

#include "openvino/pass/pass.hpp"

namespace ov {
namespace tp {

/// How TPWeightShardingPass cut one rank's weights.
struct TPShardingStats {
    size_t views = 0;  ///< contiguous slices pointing into the original buffer
    size_t copies = 0;
    size_t copied_bytes = 0;
    size_t runtime_slices = 0;
};

/// Pass 2, per rank on a clone of the analyzed model: every tp_shard node gets the rank's cut --
/// constants point at the rank's slice (a view when it is contiguous), shape constants and rt_info
/// entries get the rank's sizes, variables the rank's shape, runtime slices are inserted.
class TPWeightShardingPass : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("TPWeightShardingPass");
    TPWeightShardingPass(size_t rank, size_t world_size, std::shared_ptr<TPShardingStats> stats = nullptr)
        : m_rank(rank),
          m_world_size(world_size),
          m_stats(std::move(stats)) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    size_t m_rank;
    size_t m_world_size;
    std::shared_ptr<TPShardingStats> m_stats;
};

/// Pass 3, per rank: the collective of every tp_role projection on its output.
class TPInsertCollectives : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("TPInsertCollectives");
    TPInsertCollectives(size_t rank, size_t world_size) : m_rank(rank), m_world_size(world_size) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    size_t m_rank;
    size_t m_world_size;
};

}  // namespace tp
}  // namespace ov
