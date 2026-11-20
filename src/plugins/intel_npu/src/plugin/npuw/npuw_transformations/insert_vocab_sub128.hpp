// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::npuw {

inline constexpr const char* NPUW_SUB128_SHIFT_RT_INFO = "npuw_sub128_shift";

class InsertVocabSub128 : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("ov::npuw::InsertVocabSub128");
    InsertVocabSub128();
};

// Shifts the dequantization of the embedding Gather that reads the same u8 asymmetric vocabulary as
// `lm_head_model`. CutLMHead gives the cut-off model its own copy of the dequantization nodes, so after the
// cut the Gather in the other models has to be shifted separately. The partitioner uses one i8 closure for
// both readers only if both are shifted. Run it before InsertVocabSub128 shifts the LM head model, which
// the vocabulary is taken from.
class InsertEmbeddingSub128 : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::InsertEmbeddingSub128");

    explicit InsertEmbeddingSub128(const std::shared_ptr<ov::Model>& lm_head_model);

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    std::shared_ptr<ov::Model> m_lm_head_model;
};

class DetectVocabSharing : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::DetectVocabSharing");

    explicit DetectVocabSharing(bool& shared);

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    bool& m_shared;
};

}  // namespace ov::npuw
