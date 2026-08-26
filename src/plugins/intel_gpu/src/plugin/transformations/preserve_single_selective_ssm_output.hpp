// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::intel_gpu {

// Replaces a SelectiveSSM with its inputs when the sequence length is statically zero:
// sequence output is empty x, while recurrent state remains unchanged.
class EliminateEmptySelectiveSSM final : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("EliminateEmptySelectiveSSM");

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

// Adds a zero-copy view for a lone SelectiveSSM output so clDNN does not apply
// single-output optimizations directly to a multi-output primitive.
class PreserveSingleSelectiveSSMOutput final : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("PreserveSingleSelectiveSSMOutput");

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::intel_gpu
