// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MatMulMultiplyFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief MatMulMultiplyFusion transformation matches following graph:
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
class ov::pass::MatMulMultiplyFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MatMulMultiplyFusion", "0");
    MatMulMultiplyFusion();
};
