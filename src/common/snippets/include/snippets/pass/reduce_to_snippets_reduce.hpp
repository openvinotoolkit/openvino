// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace snippets {
namespace pass {

/**
 * @interface ReduceToSnippetsReduce
 * @brief Converts ReduceMax snd ReduceSum from openvino opset to snippets opset.
 * Also checks that reduction operation is supported by snippets.
 * @ingroup snippets
 */
class ReduceToSnippetsReduce: public ov::pass::MatcherPass {
public:
    ReduceToSnippetsReduce();
};


} // namespace pass
} // namespace snippets
} // namespace ov