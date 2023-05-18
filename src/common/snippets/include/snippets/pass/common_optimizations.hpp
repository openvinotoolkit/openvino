// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/matcher.hpp"

namespace ov {
namespace snippets {
namespace pass {

class CommonOptimizations : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("CommonOptimizations", "0");
    CommonOptimizations();
};

}  // namespace pass
}  // namespace snippets
}  // namespace ov
