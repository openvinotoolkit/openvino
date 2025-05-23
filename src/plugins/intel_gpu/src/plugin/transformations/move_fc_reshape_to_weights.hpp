// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/**
 * This transformation is applied to the FC with compressed 3D u8 weights. It moves Reshape at the weights path to the constants
 * in order to constant fold the Reshape node.
 * Example:
 *                    Weights(3D)                                            Weights(2D)
 *                       |                                                      |
 *                    Convert    Subtract_const(3D)                          Convert    Subtract_const(2D)
 *                       |      /                                               |      /
 *                   Subtract(optional)                                          Subtract(optional)
 *                       |      Multiply_const(3D)        ====>                 |      Multiply_const(2D)
 *                       |     /                                                |     /
 *                    Multiply                                               Multiply
 *                       |                                                      |
 *                    Reshape(2D)                                               |
 *                       |                                                      |
 *         Data      Transpose(optional)                       Data      Transpose(optional)
 *             \     /                                            \     /
 *         FullyConnected                                         FullyConnected
 */
class MoveFCReshapeToWeights: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MoveFCReshapeToWeights");
    MoveFCReshapeToWeights();
};

}   // namespace ov::intel_gpu
