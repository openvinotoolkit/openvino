// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

#include "snippets/op/subgraph.hpp"

namespace ov {
namespace snippets {
namespace pass {

class CommonOptimizations : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("CommonOptimizations", "0");
    CommonOptimizations();

private:
    // Move up Constants which aren't scalars from body to Subgraph and replace them with Parameters inside body
    void ExtractConstants(const std::shared_ptr<op::Subgraph>& subgraph);
    // Move up unsupported Transposes on Parameter outputs from body
    void ExtractUnsupportedTransposes(const std::shared_ptr<op::Subgraph>& subgraph);
};

}  // namespace pass
}  // namespace snippets
}  // namespace ov
