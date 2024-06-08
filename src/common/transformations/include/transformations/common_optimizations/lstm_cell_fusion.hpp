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
    OPENVINO_RTTI("LSTMCellFusionWithJointWeights", "0");
    LSTMCellFusionWithJointWeights();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief LSTMCellFusionWithSplitWeights transformation converts
 * a sequence of operations with split weights into LSTMCell op.
 */
class ov::pass::LSTMCellFusionWithSplitWeights : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("LSTMCellFusionWithSplitWeights", "0");
    LSTMCellFusionWithSplitWeights();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief LSTMCellFusion transformation replaces various sub-graphs with a LSTMCell op.
 */
class ov::pass::LSTMCellFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("LSTMCellFusion", "0");
    LSTMCellFusion() {
        add_matcher<ov::pass::LSTMCellFusionWithJointWeights>();
        add_matcher<ov::pass::LSTMCellFusionWithSplitWeights>();
    }
};
