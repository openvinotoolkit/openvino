// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace snippets {
namespace pass {

/**
 * @interface ReduceToSnippetsReduce
 * @brief Converts ngraph ReduceMax snd ReduceSum to snippets opset if this operations is supported.
 * @ingroup snippets
 */
class ReduceToSnippetsReduce: public ov::pass::MatcherPass {
public:
    ReduceToSnippetsReduce();
};


} // namespace pass
} // namespace snippets
} // namespace ov