// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>

namespace ov::intel_cpu {
class StatefulSDPAFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("StatefulSDPAFusion");
    StatefulSDPAFusion();
};

class SDPASubgraphFusion : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("SDPASubgraphFusion");

    bool run_on_model(const std::shared_ptr<ov::Model>& f) override;
};

}  // namespace ov::intel_cpu
