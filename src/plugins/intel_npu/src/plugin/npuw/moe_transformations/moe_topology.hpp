// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include "openvino/core/node.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"

namespace ov::npuw::moe {

// The semantic boundary between routing and expert execution. Selection and
// mixing are deliberately separate: selection logits may include a bias that
// must NOT be applied to the mixture weights (e.g. sigmoid/bias routers).
struct BatchedMoE {
    std::shared_ptr<ov::op::v11::TopK> topk;
    ov::Output<ov::Node> indices;
    ov::Output<ov::Node> scores;
    std::shared_ptr<ov::op::v1::Transpose> score_transpose;
    ov::Output<ov::Node> broadcast_scores;
    ov::Output<ov::Node> expert_output;
    std::shared_ptr<ov::op::v1::Multiply> weighted_output;
    std::shared_ptr<ov::op::v1::ReduceSum> reduction;
    std::shared_ptr<ov::op::v0::Tile> tile;
    ov::NodeVector expert_nodes;
    size_t num_experts = 0;
    size_t num_selected = 0;
};

// Pure analysis; never mutates the graph. Accepts independent batched expert
// FFNs with an explicit zero-based scatter of TopK indices and arbitrary mixing
// scores. Rejects cross-expert/token operations and ambiguous layouts.
std::optional<BatchedMoE> match_batched_moe(const std::shared_ptr<ov::Node>& scatter);

// Pure eligibility check shared by automatic strategy selection and lowering.
// Device routing currently requires static single-token expert computation.
bool can_device_route(const BatchedMoE& moe);

// Rebuild only the selected expert branches. Original constants/dequantization
// expressions are not modified, including when other graph users share them.
// The caller replaces the reduction only after construction succeeds.
std::shared_ptr<ov::Node> build_device_routed_moe(const BatchedMoE& moe);

}  // namespace ov::npuw::moe