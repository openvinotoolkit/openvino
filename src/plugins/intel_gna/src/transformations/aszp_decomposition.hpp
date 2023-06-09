// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "ngraph/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gna {
namespace pass {

/*
 *
 * @brief Convert a convolution node with asymmetric zero padding to concatenate zeros to the input so that the padding
 * can be symmetric: So far only supports N==1, Input to the transform node should be in NHWC The transform looks for
 * the pattern transpose->convolution->transform that get fused into one gna operation
 *
 *                                                                                   Input {N,H,W,C}
 *                                                                                          |
 *                                                                                    Reshape to 2d
 *                                                                                       {H,W*C}
 *                                                                                          |
 *                                                                      Const         Transpose 2d
 *                                                                (Zeros {W*C,Hpad})     {W*C,H}
 *                                                                        |                 |
 *                                                                        |---Concatenate---|
 *                                                                            {W*C,H+Hpad}
 *                                                                                 |
 *                                                                            Transpose 2d
 *                                                                            {H+Hpad,W*C}
 *                                                                                 |
 *                                                                           Reshape to 4d
 *                Input {N,H,W,C}                                            {N,H+Hpad,W,C}
 *                       |                                                         |
 *            Transpose to {N,C,H,W}                                  Transpose to {N,C,H+Hpad,W}
 *                       |                                                         |
 *  Convolution: pads begin: [A,B], pads end: [C,D] |------>  Convolution: pads begin: [C,B], pads end: [C,D]
 *          (for example A>C - A-C==Hpad)                                  (A now equal to C)
 *                       |                                                         |
 *             Transpose back to nhwc                                   Transpose back to nhwc
 *                       |                                                         |
 *                     output                                                    output
 *
 *                                                                                   Input {N,H,W,C}
 *                                                                                          |
 *                                                                                    Reshape to 2d
 *                                                                                       {H*W,C}
 *                                                                                          |
 *                                                                                    Transpose 2d
 *                                                                                       {C,H*W}
 *                                                                                          |
 *                                                                      Const            Reshape
 *                                                                (Zeros {C*H,Wpad})     {C*H,W}
 *                                                                        |                 |
 *                                                                        |---Concatenate---|
 *                                                                            {C*H,W+Wpad}
 *                                                                                 |
 *                                                                              Reshape
 *                                                                           {C,H*(W+Wpad)}
 *                                                                                 |
 *                                                                            Transpose 2d
 *                                                                           {H*(W+Wpad),C}
 *                                                                                 |
 *                                                                           Reshape to 4d
 *                Input {N,H,W,C}                                            {N,H,W+Wpad,C}
 *                       |                                                         |
 *            Transpose to {N,C,H,W}                                  Transpose to {N,C,H,W+Wpad}
 *                       |                                                         |
 *  Convolution: pads begin: [A,B], pads end: [C,D] |------>  Convolution: pads begin: [C,B], pads end: [C,D]
 *          (for example B>D - B-D==Wpad)                                  (A now equal to C)
 *                       |                                                         |
 *              Transpose back to nhwc                                  Transpose back to nhwc
 *                       |                                                         |
 *                     output                                                    output
 */

class AszpDecomposition : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("AszpDecomposition", "0");
    AszpDecomposition();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov