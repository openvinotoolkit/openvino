// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API LSTMCellFusion;
class TRANSFORMATIONS_API LSTMCellTfKerasFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief LSTMCellFusion transformation replaces a sequence of
 * operations with LSTMCell op.
 */
class ov::pass::LSTMCellFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("LSTMCellFusion", "0");
    LSTMCellFusion();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief LSTMCellTfKerasFusion transformation replaces a sub-graph of
 * operations with LSTMCell op. This sub-graph contains split weights
 * (input, recurrent and bias) for each gate.
 */
class ov::pass::LSTMCellTfKerasFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("LSTMCellTfKerasFusion", "0");
    LSTMCellTfKerasFusion();
};
