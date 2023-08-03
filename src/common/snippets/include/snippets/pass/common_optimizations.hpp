// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/pass/tokenization.hpp"

namespace ov {
namespace snippets {
namespace pass {

class CommonOptimizations : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("CommonOptimizations", "0");
    CommonOptimizations(const SnippetsTokenization::Config& config = {});

    // Returns True if parallelism work amount can be increased using SplitDimensionM optimization
    static bool CanOptimizeParallelWA(const std::shared_ptr<const ov::Node>& node, size_t concurrency);

private:
    // Move up Constants which aren't scalars from body to Subgraph and replace them with Parameters inside body
    void ExtractConstants(const std::shared_ptr<op::Subgraph>& subgraph);
    // Move up unsupported Transposes on Parameter outputs from body
    void ExtractUnsupportedTransposes(const std::shared_ptr<op::Subgraph>& subgraph);
    // Insert Reshape nodes after and before Parameters and Results in Subgraphs with MatMul inside
    // to split dimension M for MatMuls to increase work amount for parallelism
    // Note: works only with 3D MHA patterns
    void SplitDimensionM(const std::shared_ptr<op::Subgraph>& subgraph, size_t concurrency);
};

}  // namespace pass
}  // namespace snippets
}  // namespace ov
