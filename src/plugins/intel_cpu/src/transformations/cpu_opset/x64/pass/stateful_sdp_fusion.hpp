// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {
class StatefulSDPFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("StatefulSDPFusion", "0");
    StatefulSDPFusion();
};

class RemoveFusedAssign : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("RemoveFusedAssign");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

}   // namespace intel_cpu
}   // namespace ov
