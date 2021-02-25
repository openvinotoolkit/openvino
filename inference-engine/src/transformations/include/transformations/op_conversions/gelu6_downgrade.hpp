// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

    class TRANSFORMATIONS_API Gelu6Downgrade;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief Gelu6Downgrade converts v6::Gelu6 operation to v2::Gelu unconditionally. This is done because only limited
 * set of plugins support v6::Gelu which specifies approximation mode. For other plugins the behaviour is to use v2
 * version of the operation which does not specify the approximation mo
 */
class ngraph::pass::Gelu6Downgrade : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    Gelu6Downgrade();
};
