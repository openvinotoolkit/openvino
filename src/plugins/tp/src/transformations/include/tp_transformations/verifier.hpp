// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "openvino/pass/pass.hpp"

namespace ov {
namespace tp {

/// Proves the tp_shard / tp_role annotations of an analyzed model (global shapes) or throws naming the node.
///
/// Every output gets a state: Replicated, Split(axis, segments), Partial (the sum over ranks is the
/// value), or rank-dependent sizes (a shape vector from ShapeOf of a Split tensor or an annotated shape
/// constant). Sources: annotated constants and ReadValues, everything else Replicated. Each op type
/// has a rule; an op with a non-Replicated input and no rule, or a combination its rule rejects, fails.
class TPShardingVerifier : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("TPShardingVerifier");
    explicit TPShardingVerifier(size_t world_size) : m_world_size(world_size) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    size_t m_world_size;
};

}  // namespace tp
}  // namespace ov
