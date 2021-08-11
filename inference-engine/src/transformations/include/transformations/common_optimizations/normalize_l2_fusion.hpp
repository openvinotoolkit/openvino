// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>
#include <memory>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "ngraph/pattern/matcher.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API NormalizeL2Fusion;
class TRANSFORMATIONS_API NormalizeL2FusionWithMax;
class TRANSFORMATIONS_API NormalizeL2FusionWithAdd;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief NormalizeL2FusionWithMax transformation replaces a sub-graph
 * x/(max(sqrt(sum(x[j0, ..., jN]**2), eps)) with a NormalizeL2 op.
 */
class ov::pass::NormalizeL2FusionWithMax: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    NormalizeL2FusionWithMax();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief NormalizeL2FusionWithAdd transformation replaces a sub-graph
 * x/(add(sqrt(sum(x[j0, ..., jN]**2), eps)) with a NormalizeL2 op.
 */
class ov::pass::NormalizeL2FusionWithAdd: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    NormalizeL2FusionWithAdd();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief NormalizeL2Fusion transformation replaces various sub-graphs with a NormalizeL2 op.
 */
class ov::pass::NormalizeL2Fusion: public ov::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    NormalizeL2Fusion() {
        add_matcher<ov::pass::NormalizeL2FusionWithMax>();
        add_matcher<ov::pass::NormalizeL2FusionWithAdd>();
    }
};
