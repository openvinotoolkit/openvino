// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API SwishFusion;
class TRANSFORMATIONS_API SwishFusionWithSigmoid;
class TRANSFORMATIONS_API SwishFusionWithSigmoidWithBeta;
class TRANSFORMATIONS_API SwishFusionWithBeta;
class TRANSFORMATIONS_API SwishFusionWithoutBeta;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief SwishFusionWithSigmoid replaces a sub-graphs x * Sigmoid(x) with a Swish op.
 */
class ov::pass::SwishFusionWithSigmoid : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("SwishFusionWithSigmoid");
    SwishFusionWithSigmoid();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief SwishFusionWithSigmoid replaces a sub-graphs x * Sigmoid(x * beta) with a Swish op.
 */
class ov::pass::SwishFusionWithSigmoidWithBeta : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("SwishFusionWithSigmoidWithBeta");
    SwishFusionWithSigmoidWithBeta();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief SwishFusionWithSigmoid replaces a sub-graphs x / (1.0 + exp(-x * beta)) with a Swish op.
 */
class ov::pass::SwishFusionWithBeta : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("SwishFusionWithBeta");
    SwishFusionWithBeta();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief SwishFusionWithSigmoid replaces a sub-graphs x / (1.0 + exp(-x)) with a Swish op.
 */
class ov::pass::SwishFusionWithoutBeta : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("SwishFusionWithoutBeta");
    SwishFusionWithoutBeta();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief SwishFusion transformation replaces various sub-graphs with a Swish op.
 */
class ov::pass::SwishFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("SwishFusion");
    SwishFusion() {
        add_matcher<ov::pass::SwishFusionWithSigmoid>();
        add_matcher<ov::pass::SwishFusionWithSigmoidWithBeta>();
        add_matcher<ov::pass::SwishFusionWithBeta>();
        add_matcher<ov::pass::SwishFusionWithoutBeta>();
    }
};
