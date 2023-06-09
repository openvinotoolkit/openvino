// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gna {
namespace pass {

/**
 * @brief Reduce the rank of Transpose shape by fusing by fusing dimentions
 *    [A, B, C, D]            [A, B, C, D]
 *         |                       |
 *     Transpose                Reshape
 *         |                       |
 *    [A, D, B, C]             [A, B*C, D]
 *         |                       |
 *         |                   Transpose
 *         |           ->          |
 *         |           <-     [A, D, B*C]
 *         |                       |
 *         |                    Reshape
 *         |                       |
 *         |                  [A, D, B, C]
 *         |                       |
 *     Any Layer                Any Layer
 */
class Transpose2D : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("Transpose2D", "0");
    Transpose2D();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
