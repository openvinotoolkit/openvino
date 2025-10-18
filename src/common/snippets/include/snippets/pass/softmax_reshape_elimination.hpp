// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "snippets/snippets_visibility.hpp"

namespace ov::snippets::pass {

/**
 * @interface SoftmaxReshapeElimination
 * @brief The pass removes Reshape operations around Softmax if possible
 * @ingroup snippets
 */
class SNIPPETS_API SoftmaxReshapeElimination : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("snippets::pass::SoftmaxReshapeElimination");
    SoftmaxReshapeElimination();
};

}  // namespace ov::snippets::pass
