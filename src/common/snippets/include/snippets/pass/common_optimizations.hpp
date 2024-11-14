// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "snippets/pass/tokenization.hpp"

namespace ov {
namespace snippets {
namespace pass {

class CommonOptimizations : public ov::pass::MatcherPass {
    class SubgraphPass;
    class SubgraphManager;
    friend class ExtractConstants;
    friend class ExtractUnsupportedTransposes;
    friend class SplitDimensionM;

public:
    OPENVINO_RTTI("CommonOptimizations", "0");
    CommonOptimizations(const SnippetsTokenization::Config& config);
};

}  // namespace pass
}  // namespace snippets
}  // namespace ov
