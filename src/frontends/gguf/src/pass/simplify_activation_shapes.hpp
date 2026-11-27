// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::frontend::gguf::pass {

// Remove GGML's singleton activation axis through elementwise and normalization paths,
// preserving layout at MatMul boundaries and fusion-sensitive RMS/RoPE paths.
class SimplifyActivationShapes : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::frontend::gguf::pass::SimplifyActivationShapes");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::frontend::gguf::pass
