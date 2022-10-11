// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

/**
 * @interface TransofrmConvertToConvertTruncation
 * @brief Transform Convert to ConvertTruncation with specification conversion rules
 *        Note: ConvertTruncation op is covered by specification of "Convert" op
 *              This op is used for real Convert ops inside subgraph body in CPU Plugin
 * @ingroup snippets
 */
class TransformConvertToConvertTruncation: public ngraph::pass::MatcherPass {
public:
    TransformConvertToConvertTruncation();
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
