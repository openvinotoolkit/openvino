// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "snippets/pass/tokenization.hpp"

namespace ov {
namespace snippets {
namespace pass {

class CommonOptimizations : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("CommonOptimizations", "0");
    CommonOptimizations(const SnippetsTokenization::Config& config = {});
};

}  // namespace pass
}  // namespace snippets
}  // namespace ov
