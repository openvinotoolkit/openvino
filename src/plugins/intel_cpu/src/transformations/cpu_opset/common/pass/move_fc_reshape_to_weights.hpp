// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {

/**
 * This transformation is applied to the FC with compressed 3D u8 weights. It moves Reshape at the weights path to the constants
 * in order to constant fold the Reshape node.
 * Example:
 *                    Weights(3D)                                            Weights(2D)
 *                       |                                                      |
 *                    Convert    Subtract_const(3D)                          Convert    Subtract_const(2D)
 *                       |      /                                               |      /
 *                   Subtract(opt)                                          Subtract(opt)
 *                       |      Multiply_const(3D)        ====>                 |      Multiply_const(2D)
 *                       |     /                                                |     /
 *                    Multiply                                               Multiply
 *                       |                                                      |
 *                    Reshape(2D)                                               |
 *                       |                                                      |
 *         Data      Transpose(opt)                               Data      Transpose(opt)
 *             \     /                                                \     /
 *         FullyConnected                                         FullyConnected
 */
class MoveFCReshapeToWeights: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MoveFCReshapeToWeights", "0");
    MoveFCReshapeToWeights();
};

}   // namespace intel_cpu
}   // namespace ov
