// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <unordered_set>

#include "openvino/core/model.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::intel_cpu {
class StatefulSDPAFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("StatefulSDPAFusion");
    StatefulSDPAFusion();

private:
    // Variable ids of past_k / past_v whose SDPA has already been fused; later matches on the
    // same Variable are reader-only SDPAs and must be skipped.
    std::unordered_set<std::string> m_processed_k_variable_ids;
    std::unordered_set<std::string> m_processed_v_variable_ids;
};

class SDPASubgraphFusion : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("SDPASubgraphFusion");

    bool run_on_model(const std::shared_ptr<ov::Model>& f) override;
};

}  // namespace ov::intel_cpu
