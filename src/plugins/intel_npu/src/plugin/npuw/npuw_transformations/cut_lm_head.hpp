// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/pass/pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"

namespace ov::npuw {

namespace patterns {

// Ops that may follow the LM-head MatMul before the logits Result:
//   MatMul -> Result
//   MatMul -> Add | Transpose | Convert -> Result
//   MatMul -> Multiply | Divide -> Result                     (logits scaling, Granite-4.0-h-micro)
//   MatMul -> Multiply | Divide -> Tanh -> Multiply -> Result (logits softcapping)
// CutLMHead, InsertVocabSub128 and DetectVocabSharing use this one definition, so that all of
// them accept the same LM heads.
struct LMHeadTerminal {
    explicit LMHeadTerminal(const std::shared_ptr<ov::Node>& matmul);

    // The matched op that feeds the Result: the terminal op, or the MatMul if there is none.
    std::shared_ptr<ov::Node> last_op(const ov::pass::pattern::PatternValueMap& values) const;

    std::shared_ptr<ov::Node> matmul;
    std::shared_ptr<ov::Node> add;
    std::shared_ptr<ov::Node> transpose;
    std::shared_ptr<ov::Node> convert;
    std::shared_ptr<ov::Node> scale;    // Multiply | Divide
    std::shared_ptr<ov::Node> softcap;  // Multiply after Tanh(scale)
    std::shared_ptr<ov::Node> output;   // Or of the MatMul and all terminal ops: the Result input
};

}  // namespace patterns

// Splits the LM head (vocabulary MatMul) off the input model and returns
// it as a separate one.
// The cut point is the first input of the matched MatMul. The matched logits
// Result is repurposed as the output-embeddings Result of the original model.
// The returned model has independent nodes, including tied-weight dequantization
// branches; constant data buffers remain shared.
class CutLMHead : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::CutLMHead");
    explicit CutLMHead(std::shared_ptr<ov::Model>& lm_head_model);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    std::shared_ptr<ov::Model>& m_lm_head_model;
};

}  // namespace ov::npuw
