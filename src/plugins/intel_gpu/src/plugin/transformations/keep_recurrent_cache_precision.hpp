// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "openvino/pass/pass.hpp"
namespace ov::intel_gpu {
// Qwen3.5 llama.cpp fused recurrence reads cache precision directly and performs
// computation-precision rounding inside its kernel, avoiding full-cache conversions.
class KeepRecurrentCachePrecision : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::intel_gpu::KeepRecurrentCachePrecision");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};
}  // namespace ov::intel_gpu
