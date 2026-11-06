// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/weights_prefetch.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/// @brief Scheduler slot shared between the fusion pass and its matcher.
struct WeightsPrefetchSlot {
    std::unique_ptr<ov::util::OrderedPrefetchScheduler> scheduler;
};

class FullyConnectedHorizontalFusionMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FullyConnectedHorizontalFusionMatcher");
    explicit FullyConnectedHorizontalFusionMatcher(bool fuse_mlp_swiglu = false,
                                                   std::shared_ptr<WeightsPrefetchSlot> prefetch = nullptr);
};

/// @brief Fuses the FullyConnectedCompressed ops sharing one input.
class FullyConnectedHorizontalFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("FullyConnectedHorizontalFusion");
    explicit FullyConnectedHorizontalFusion(bool fuse_mlp_swiglu = false, bool prefetch_weights = false);

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    std::shared_ptr<WeightsPrefetchSlot> m_prefetch;
    bool m_prefetch_weights = false;
    size_t m_min_num_fcs_to_fuse = 3;
};

}   // namespace ov::intel_gpu
