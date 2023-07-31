// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gna {
namespace pass {

/**
 * @brief Replace the unsupported 2D Transpose by sequence of supported Transposes.
 *
 *       [A, B]                  [A, B]
 *         |                       |
 *      Transpose                Reshape
 *         |                       |
 *       [B, A]                 [A1, B1]
 *         |                       |
 *         |                   Transpose
 *         |           ->          |
 *         |           <-       [B1, A1]
 *         |                       |
 *         |                    Reshape
 *         |                       |
 *         |                    [A2, B2]
 *         |                       |
 *         |                   Transpose
 *         |                       |
 *         |                    [B2, A2]
 *         |                       |
 *         |                    Reshape
 *         |                       |
 *         |                     [B, A]
 *         |                       |
 *     Any Layer                Any Layer
 */
class ReplaceBigTranspose : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ReplaceBigTranspose", "0");
    ReplaceBigTranspose();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
