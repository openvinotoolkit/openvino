// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/op/constant.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::npuw {

inline constexpr const char* NPUW_SUB128_SHIFT_RT_INFO = "npuw_sub128_shift";

// Constants of a quantized vocabulary table.
struct Vocab {
    std::shared_ptr<ov::op::v0::Constant> weight;
    std::shared_ptr<ov::op::v0::Constant> zerop;  // nullptr if absent
    std::shared_ptr<ov::op::v0::Constant> scale;  // nullptr if absent
};

// Shifts the u8 asymmetric vocabulary of the LM head by 128 (Sub128). If `shifted` is given, the
// vocabularies that were shifted are added to it.
class InsertVocabSub128 : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("ov::npuw::InsertVocabSub128");
    InsertVocabSub128();
    explicit InsertVocabSub128(std::vector<Vocab>& shifted);
};

// Shifts the embedding Gather that reads one of `lm_head_vocabs`, with the same dequantization pattern as
// InsertVocabSub128. CutLMHead gives the LM head model its own copy of the dequantization nodes, so the
// Gather in the other models has to be shifted separately. Partitioning uses one i8 closure for the
// embedding and the LM head only if both are shifted; otherwise the u8 table stays next to the i8 one and
// no memory is saved.
class InsertEmbeddingSub128 : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("ov::npuw::InsertEmbeddingSub128");
    explicit InsertEmbeddingSub128(const std::vector<Vocab>& lm_head_vocabs);
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
