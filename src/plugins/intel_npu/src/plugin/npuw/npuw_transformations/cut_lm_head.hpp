// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/pass/pass.hpp"

namespace ov::npuw {

// Splits the LM head (vocabulary MatMul) off the input model and returns
// it as a separate one.
// The cut point is the first input of the matched MatMul. The matched logits
// Result is repurposed as the output-embeddings Result of the original model.
class CutLMHead : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::CutLMHead");
    explicit CutLMHead(std::shared_ptr<ov::Model>& lm_head_model);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    std::shared_ptr<ov::Model>& m_lm_head_model;
};

}  // namespace ov::npuw
