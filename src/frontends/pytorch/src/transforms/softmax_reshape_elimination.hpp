// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/matcher.hpp"

namespace ov::frontend::pytorch::pass {

/**
 * @interface SoftmaxReshapeElimination
 * @brief The pass removes Reshape operations around Softmax if possible
 * @ingroup snippets
 */
class SoftmaxReshapeElimination : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::pytorch::pass::SoftmaxReshapeElimination");
    SoftmaxReshapeElimination();
};

}  // namespace ov::frontend::pytorch::pass
