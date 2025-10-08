// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "snippets/snippets_visibility.hpp"

namespace ov::snippets::pass {

/**
 * @interface ReduceToSnippetsReduce
 * @brief Converts ReduceMax snd ReduceSum from openvino opset to snippets opset.
 * Also checks that reduction operation is supported by snippets.
 * @ingroup snippets
 */
class SNIPPETS_API ReduceToSnippetsReduce : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("snippets::pass::ReduceToSnippetsReduce");
    ReduceToSnippetsReduce();
};

}  // namespace ov::snippets::pass
