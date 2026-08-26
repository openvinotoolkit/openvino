// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::intel_gpu {

// Keeps standalone SelectiveSSM data and metadata in their original precision.
// Paged Attention models do not register this pass and continue to use inference/cache precision.
class PreserveStandaloneSelectiveSSMPrecision final : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("PreserveStandaloneSelectiveSSMPrecision");

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

// Restores the mutable state table precision after ConvertPagedAttnInputs. This pass is
// registered only for standalone models, before the transformation manager validates them.
class RestoreStandalonePagedSelectiveSSMStatePrecision final : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("RestoreStandalonePagedSelectiveSSMStatePrecision");

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::intel_gpu
