// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MatmulSplitDecomposition;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief MatmulSplitDecomposition transformation matches following graph:
 *
 *         +----------+            +----------+
 *         |    A     |            |    B     |
 *         +----------+            +----------+
 *              |                       |
 *              -----------    ----------
 *                        |    |
 *                        v    v
 *                      +--------+
 *                      | MatMul |
 *                      +--------+
 *                          |
 *                          v
 *                     +----------+     +----------+
 *                     | Multiply |<----| Constant |
 *                     +----------+     +----------+
 *
 *
 * and replaces with:
 *
 *                           +-------+   +----------+
 *                           |   B   |   | Constant |
 *                           +-------+   +----------+
 *                                |            |
 *                                ------  ------
 *                                     |  |
 *                                     v  v
 *         +----------+            +----------+
 *         |    A     |            | Multiply |
 *         +----------+            +----------+
 *              |                       |
 *              -----------    ----------
 *                        |    |
 *                        v    v
 *                      +--------+
 *                      | MatMul |
 *                      +--------+
 */
class ov::pass::MatmulSplitDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MatmulSplitDecomposition", "0");
    MatmulSplitDecomposition();
    void split_weights(const Output<Node>& weights, NodeVector& new_weights,
                       const Output<Node>& bias, NodeVector& new_bias);
};
