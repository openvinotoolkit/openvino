// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/weights_prefetch.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/// @brief Fuses the FullyConnectedCompressed ops sharing one input.
/// Register it with pass::Manager::register_pass only. Do not nest it via GraphRewrite::add_matcher<T>():
/// that copies the matchers and destroys this pass, while the matcher callback still points to its m_scheduler.
class FullyConnectedHorizontalFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("FullyConnectedHorizontalFusion");
    explicit FullyConnectedHorizontalFusion(bool fuse_mlp_swiglu = false, bool prefetch_weights = false);

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    std::unique_ptr<ov::util::OrderedPrefetchScheduler> m_scheduler;  // Set only while run_on_model() runs.
    bool m_prefetch_weights = false;
    size_t m_min_num_fcs_to_fuse = 3;
};

}   // namespace ov::intel_gpu
