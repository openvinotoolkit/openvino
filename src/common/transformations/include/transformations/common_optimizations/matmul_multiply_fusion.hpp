// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API MatMulMultiplyFusion;

}  // namespace pass
}  // namespace ngraph

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
class ngraph::pass::MatMulMultiplyFusion : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("MatMulMultiplyFusion", "0");
    MatMulMultiplyFusion();
};
