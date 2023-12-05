// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_gna {
namespace pass {

/**
 * @brief Brodcast data in Const layer
 * Transformation recognizes the following patterns
 *
 * Constant    Any
 *       |     |
 *       Eltwise
 *
 * Constant
 *    |
 * FakeQuantize     Any
 *            |     |
 *            Eltwise
 *
 * Where Eltwise node is one of the: Multiply, Substract, Add or ScaleShiftIE
 * There are different types of broadcasting: NONE/EXPLICIT, NUMPY and PDPD
 *
 * If eltwise node inputs have different shapes and one the inputs is Constant node
 * we can update (broadcast) Constant to have the same shape as another input.
 * Transformation doesn't modify graph structure, but modifies Constant
 * examples:
 *
 * NUMPY broadcasting
 * Eltwise non-constant shape | Constant prev shape/values | Constant new shape/values
 *         {3,2}                       {2}/{1,2}             {3,2}/{1, 2, 1, 2, 1, 2}
 *         {2,3}                       {2,1}/{1,2}           {2,3}/{1, 1, 1, 2, 2, 2}
 *
 * PDPD broadcasting
 * Eltwise non-constant shape | Constant prev shape/values | Constant new shape/values
 *         {3,2}                       {3, 1}/{1,2,3}             {3,2}/{1, 1, 2, 2, 3, 3}
 *
 * NONE/EXPLICIT broadcasting doesn't support broadcasting at all
 *
 * For information about broadcasting rules see
 * https://github.com/openvinotoolkit/openvino/blob/master/docs/ops/broadcast_rules.md
 */
class BroadcastAddMultiplyConst : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("BroadcastAddMultiplyConst", "0");
    BroadcastAddMultiplyConst();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
