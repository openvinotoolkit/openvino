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

class DetectVocabSharing : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::DetectVocabSharing");

    explicit DetectVocabSharing(bool& shared);

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    bool& m_shared;
};

}  // namespace ov::npuw
