// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::intel_gpu {

class MarkBatchedNmsStaticClassCount : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::intel_gpu::MarkBatchedNmsStaticClassCount");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

// Replaces the lowered PyTorch-style batched NMS pattern with a real
// MulticlassNms when the graph still exposes the original 1D selected indices
// output. This removes the artificial coordinate shift used to emulate
// per-class suppression with a single NonMaxSuppression call.
class ConvertBatchedNmsToMulticlassNms : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::intel_gpu::ConvertBatchedNmsToMulticlassNms");
    ConvertBatchedNmsToMulticlassNms();
};

}  // namespace ov::intel_gpu

