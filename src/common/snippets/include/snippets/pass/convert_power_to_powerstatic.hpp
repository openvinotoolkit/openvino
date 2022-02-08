// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

/**
 * @interface ConvertConstantsToScalars
 * @brief Replace Power with a scalar input with snippets::op::PowerStatic for generation of a more optimal code.
 * @ingroup snippets
 */
class ConvertPowerToPowerStatic: public ngraph::pass::MatcherPass {
public:
    ConvertPowerToPowerStatic();
};

} // namespace pass
} // namespace snippets
} // namespace ngraph