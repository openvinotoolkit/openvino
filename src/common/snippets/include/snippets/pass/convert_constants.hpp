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
 * @brief Replace only constants which are should be represented as scalars during code generation.
 *        Only single-value (0D) constants are currently supported.
 * @ingroup snippets
 */
class ConvertConstantsToScalars: public ngraph::pass::MatcherPass {
public:
    ConvertConstantsToScalars();
};

/**
 * @interface ConvertConstantsToParameters
 * @brief Move up Constants which aren't scalars from body to Subgraph
 *        and replace them with Parameters inside body
 * @ingroup snippets
 */
class ConvertConstantsToParameters : public ngraph::pass::MatcherPass {
public:
    ConvertConstantsToParameters();
};

} // namespace pass
} // namespace snippets
} // namespace ngraph
