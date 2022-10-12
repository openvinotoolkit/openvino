// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <utility>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API PReluFusion;
class TRANSFORMATIONS_API PReluFusionNegativeAdd;
class TRANSFORMATIONS_API PReluFusionNegativeSub;
class TRANSFORMATIONS_API PReluFusionMultiplyAdd;
class TRANSFORMATIONS_API PReluFusionMultiplySub;

}  // namespace pass
}  // namespace ov

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
class ov::pass::PReluFusionNegativeAdd : public ov::pass::MatcherPass {
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
class ov::pass::PReluFusionNegativeSub : public ov::pass::MatcherPass {
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
class ov::pass::PReluFusionMultiplyAdd : public ov::pass::MatcherPass {
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
class ov::pass::PReluFusionMultiplySub : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    PReluFusionMultiplySub();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief PReluFusion transformation replaces various sub-graphs with a PRelu op.
 */
class ov::pass::PReluFusion : public ov::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    PReluFusion() {
        add_matcher<ov::pass::PReluFusionNegativeAdd>();
        add_matcher<ov::pass::PReluFusionNegativeSub>();
        add_matcher<ov::pass::PReluFusionMultiplyAdd>();
        add_matcher<ov::pass::PReluFusionMultiplySub>();
    }
};

namespace ngraph {
namespace pass {
using ov::pass::PReluFusion;
using ov::pass::PReluFusionMultiplyAdd;
using ov::pass::PReluFusionMultiplySub;
using ov::pass::PReluFusionNegativeAdd;
using ov::pass::PReluFusionNegativeSub;
}  // namespace pass
}  // namespace ngraph
