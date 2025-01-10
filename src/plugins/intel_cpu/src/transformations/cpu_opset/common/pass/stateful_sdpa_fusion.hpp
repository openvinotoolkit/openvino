// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {
class StatefulSDPAFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("StatefulSDPAFusion");
    StatefulSDPAFusion();
};

class SDPASubgraphFusion : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("SDPASubgraphFusion", "0");

    bool run_on_model(const std::shared_ptr<ov::Model>& f) override;
};

}  // namespace intel_cpu
}  // namespace ov
