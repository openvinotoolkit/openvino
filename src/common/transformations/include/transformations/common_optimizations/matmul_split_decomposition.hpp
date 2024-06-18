// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MatmulSplitDecomposition;
class TRANSFORMATIONS_API MatmulVariadicSplitDecomposition;

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
    void split_weights(const Output<Node>& weights, OutputVector& new_weights,
                       const Output<Node>& bias, OutputVector& new_bias);
};

class ov::pass::MatmulVariadicSplitDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MatmulVariadicSplitDecomposition", "0");
    MatmulVariadicSplitDecomposition();
    void split_weights(const Output<Node>& weights, OutputVector& new_weights,
                       const Output<Node>& split_length);
};