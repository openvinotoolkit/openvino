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
class TRANSFORMATIONS_API PReluFusionAbsSubMulMulAdd;

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
    OPENVINO_RTTI("PReluFusionNegativeAdd", "0");
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
    OPENVINO_RTTI("PReluFusionNegativeSub", "0");
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
    OPENVINO_RTTI("PReluFusionMultiplyAdd", "0");
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
    OPENVINO_RTTI("PReluFusionMultiplySub", "0");
    PReluFusionMultiplySub();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief PReluFusionAbsSubMulMulAdd transformation replaces a sub-graph
 *            Op
 *          /  |  \
 *        Relu |  Abs
 *         |    \  |
 *         |      Sub
 *         |       |
 *         |    Multiply
 *         |       |
 *         |    Multiply (0.5)
 *          \     /
 *            Add
 */
class ov::pass::PReluFusionAbsSubMulMulAdd : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("PReluFusionAbsSubMulMulAdd", "0");
    PReluFusionAbsSubMulMulAdd();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief PReluFusion transformation replaces various sub-graphs with a PRelu op.
 */
class ov::pass::PReluFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("PReluFusion", "0");
    PReluFusion() {
        add_matcher<ov::pass::PReluFusionNegativeAdd>();
        add_matcher<ov::pass::PReluFusionNegativeSub>();
        add_matcher<ov::pass::PReluFusionMultiplyAdd>();
        add_matcher<ov::pass::PReluFusionMultiplySub>();
        add_matcher<ov::pass::PReluFusionAbsSubMulMulAdd>();
    }
};
