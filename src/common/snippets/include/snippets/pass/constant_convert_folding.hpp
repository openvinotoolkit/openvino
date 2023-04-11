// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

/**
 * @interface ConstantConvertFolding
 * @brief Constant folding only for branches with Constant and Convert
 * @ingroup snippets
 */
class ConstantConvertFolding: public ngraph::pass::MatcherPass {
public:
    ConstantConvertFolding();
};


} // namespace pass
} // namespace snippets
} // namespace ngraph