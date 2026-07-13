// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::npuw {

/// Detection-only pass: returns true if the model contains a pure causal
/// (lower-triangular) attention mask pattern — i.e. a LessEqual(K_range, Q_range)
/// that is NOT combined with a sliding-window bound.

class DetectCausalMask : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::DetectCausalMask");
    DetectCausalMask() = default;
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::npuw
