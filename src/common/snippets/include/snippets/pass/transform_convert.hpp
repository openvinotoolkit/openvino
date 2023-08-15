// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/matcher.hpp"

namespace ov {
namespace snippets {
namespace pass {

/**
 * @interface TransformConvertToConvertTruncation
 * @brief Transform Convert to ConvertTruncation with specification conversion rules
 *        Note: ConvertTruncation op is covered by specification of "Convert" op
 *              This op is used for real Convert ops inside subgraph body in CPU Plugin
 * @ingroup snippets
 */
class TransformConvertToConvertTruncation: public ov::pass::MatcherPass {
public:
    TransformConvertToConvertTruncation();
};

}  // namespace pass
}  // namespace snippets
}  // namespace ov
