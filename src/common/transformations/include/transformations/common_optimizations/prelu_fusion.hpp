// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <utility>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API PReluFusion;
class TRANSFORMATIONS_API PReluFusionNegativeAdd;
class TRANSFORMATIONS_API PReluFusionNegativeSub;
class TRANSFORMATIONS_API PReluFusionMultiplyAdd;
class TRANSFORMATIONS_API PReluFusionMultiplySub;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief PReluFusionNegativeAdd transformation replaces a sub-graph
 *             Op
 *          /     \
 *        Relu  Negative
 *         |       |
 *         |      Relu
 *         |       |
 *         |    Negative
 *         |       |
 *         |    Multiply
 *          \     /
 *            Add
 */
class ngraph::pass::PReluFusionNegativeAdd : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    PReluFusionNegativeAdd();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief PReluFusionNegativeSub transformation replaces a sub-graph
 *             Op
 *          /     \
 *        Relu  Negative
 *         |       |
 *         |      Relu
 *         |       |
 *         |    Multiply
 *          \     /
 *            Sub
 */
class ngraph::pass::PReluFusionNegativeSub : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    PReluFusionNegativeSub();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief PReluFusionMultiplyAdd transformation replaces a sub-graph
 *             Op
 *          /     \
 *        Relu  Multiply (-1)
 *         |       |
 *         |      Relu
 *         |       |
 *         |    Multiply
 *          \     /
 *            Add
 */
class ngraph::pass::PReluFusionMultiplyAdd : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    PReluFusionMultiplyAdd();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief PReluFusionMultiplySub transformation replaces a sub-graph
 *             Op
 *          /     \
 *        Relu  Multiply (-1)
 *         |       |
 *         |      Relu
 *         |       |
 *         |    Multiply
 *          \     /
 *            Sub
 */
class ngraph::pass::PReluFusionMultiplySub : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    PReluFusionMultiplySub();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief PReluFusion transformation replaces various sub-graphs with a PRelu op.
 */
class ngraph::pass::PReluFusion : public ngraph::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    PReluFusion() {
        add_matcher<ngraph::pass::PReluFusionNegativeAdd>();
        add_matcher<ngraph::pass::PReluFusionNegativeSub>();
        add_matcher<ngraph::pass::PReluFusionMultiplyAdd>();
        add_matcher<ngraph::pass::PReluFusionMultiplySub>();
    }
};