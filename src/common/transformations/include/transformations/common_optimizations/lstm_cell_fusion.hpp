// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API LSTMCellFusion;
class TRANSFORMATIONS_API LSTMCellFusionWithJointWeights;
class TRANSFORMATIONS_API LSTMCellFusionWithSplitWeights;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief LSTMCellFusionWithJointWeights transformation converts
 * a sequence of operations with merged weights into LSTMCell op.
 */
class ov::pass::LSTMCellFusionWithJointWeights : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("LSTMCellFusionWithJointWeights");
    LSTMCellFusionWithJointWeights();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief LSTMCellFusionWithSplitWeights transformation converts
 * a sequence of operations with split weights into LSTMCell op.
 */
class ov::pass::LSTMCellFusionWithSplitWeights : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("LSTMCellFusionWithSplitWeights");
    LSTMCellFusionWithSplitWeights();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief LSTMCellFusion transformation replaces various sub-graphs with a LSTMCell op.
 */
class ov::pass::LSTMCellFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("LSTMCellFusion");
    LSTMCellFusion() {
        add_matcher<ov::pass::LSTMCellFusionWithJointWeights>();
        add_matcher<ov::pass::LSTMCellFusionWithSplitWeights>();
    }
};
