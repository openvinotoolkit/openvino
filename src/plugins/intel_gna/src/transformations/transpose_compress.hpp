// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gna {
namespace pass {

/**
 * @brief Reduce the rank of Transpose shape by merging consecutive dimensions
 * if those dimensions are not changed by transposition order.
 * For example, shape[2,3,4] -> Transpose[2, 0, 1] -> shape[4,2,3] will be replaced with
 *              shape[6,4]   -> Transpose[1, 0]    -> shape[4,6]
 * If the new Transpose layer is not supported by GNA (see full conditions in the is_transpose_supported())
 * then Transpose will not be changed.
 *
 *    [A, B, C, D]            [A, B, C, D]
 *         |                       |
 *    Transpose(0, 3, 1, 2)     Reshape
 *         |                       |
 *    [A, D, B, C]             [A, B*C, D]
 *         |                       |
 *         |                   Transpose(0, 2, 1)
 *         |           ->          |
 *         |           <-     [A, D, B*C]
 *         |                       |
 *         |                    Reshape
 *         |                       |
 *         |                  [A, D, B, C]
 *         |                       |
 *     Any Layer                Any Layer
 */
class TransposeCompress : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("TransposeCompress", "0");
    TransposeCompress();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
