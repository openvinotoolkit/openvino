// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MatmulGatherDecomposition;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief MatmulGatherDecomposition transformation matches following graph:
 *
 *         +----------+
 *         |  input   |
 *         +----------+
 *              |
 *              v
 *         +----------+
 *         |  MatMul  |
 *         +----------+
 *              |
 *              v
 *         +------------+
 *         | Some nodes |
 *         +------------+
 *              |
 *              v
 *         +-----------------------+
 *         |       Transpose       |
 *         +-----------------------+
 *          |          |          |
 *          v          v          v
 *     +-------+   +-------+   +-------+
 *     |Gather |   |Gather |   |Gather |
 *     +-------+   +-------+   +-------+
 * and replaces with:
 *
 *         +-----------------------+
 *         |       input           |
 *         +-----------------------+
 *          |          |          |
 *          v          v          v
 *     +-------+   +-------+   +-------+
 *     |MatMul |   |MatMul |   |MatMul |
 *     +-------+   +-------+   +-------+
 *          |          |          |
 *          v          v          v
 *     +-------+   +-------+   +-------+
 *     |Nodes  |   |Nodes  |   |Nodes  |
 *     +-------+   +-------+   +-------+
 *          |          |          |
 *          v          v          v
 *   +---------+  +---------+  +---------+
 *   |Transpose|  |Transpose|  |Transpose|
 *   +---------+  +---------+  +---------+
 */
class ov::pass::MatmulGatherDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MatmulGatherDecomposition", "0");
    MatmulGatherDecomposition();
    void split_weights(const Output<Node>& weights,
                       OutputVector& new_weights,
                       Output<Node>* bias,
                       OutputVector& new_bias,
                       const bool& transpos_b);
};