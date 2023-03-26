// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_gna {
namespace pass {


/**
 * @brief Convert a Add Operation to a Group convolution followed by Add with NCHW layout:
 *                                    
 * Add                                                                                       * Add:  Input1: [B, A]                                            *Add:  Input1: [A,B]
 * Input1: [A,B]                                                                             *       Input2: [B, 1]                                            *      Input2: [A,B]
 * Input2: [1,B]                                  Input: [A, B]                              *       Output: [B, A]                                            *      Output: [A,B]  
 * Output: [A,B]                                    |                                                                                                       
 *                                            Transpose -> [B, A]                                   Input: [B, A]                                                  Input: [A, B]           
 *                                                  |                                                    |                                                               |
 *                                          Reshape 4D [1, B, 1, A]                             Reshape 4D [1, B, 1, A]                                       Reshape 4D [1, A*B, 1, 1]
 *                                                  |                                                    |                                                               |
 *                                   Group Convolution in NCHW layout                        Group Convolution in NCHW layout                               Group Convolution in NCHW layout
 *                                      Input:  [1, B, 1, A]                                    Input:  [1, B, 1, A]                                          Input:  [1, A*B, 1, 1]
 *                                      Kernel: [B, 1, 1, 1, 1] Initlialize to 1.0f             Kernel: [B, 1, 1, 1, 1] Initlialize to 1.0f                   Kernel: [A*B, 1, 1, 1, 1] Initlialize to 1.0f
 *                                         Output: [1, B, 1, A]                                    Output: [1, B, 1, A]                                          Output: [1, A*B, 1, 1]
 *                                                  |                                                    |                                                                |
 *                                                 Add                                                  Add                                                              Add
 *                                      I0:  [1, B, 1, A]                                    I0:  [1, B, 1, A]                                                    I0:  [1, A*B, 1, ]   
 *                                      I1:  [1, B, 1, 1] Assign Input2                      I1:  [1, B, 1, 1] Assign Input2                                      I1:  [1, A*B, 1, 1] Assign Input2
 *                                      Output: [1, B, 1, A]                                    Output: [1, B, 1, A]                                              Output: [1, A*B, 1, 1]
 *                                                  |                                                    |                                                                |
 *                                          Reshape 2D  [B, A]                                      Reshape 2D  [B, A]                                             Reshape 2D  [A, B]
 *                                                  |
 *                                          Transpose -> [A, B]
 *                                                  
 *                                      
 */
class AddDecomposition : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("AddDecomposition", "0");
    AddDecomposition();
};

/**
 * @brief Convert a Subtract operation a Group convolution followed by Add with NCHW layout:
 *
 *
 *                                            Input: [A, B]
 *                                                  |
 *                                           Transpose -> [B, A]
 *                                                  |
 *                                             Reshape 4D
 *                                                  |
 * Subtract                            Group Convolution in NCHW layout
 * Input1: [A, B]            ------->    Input:  [1, B, 1, A]
 * Input2: [1, B]                       Kernel: [B, 1, 1, 1, 1] Initlialize to -1.0f
 * Output: [A, B]                       Output: [1, B, 1, A]
 *                                                  |
 *                                                 Add
 *                                      I0:  [1, B, 1, 1]
 *                                      I1:  [1, B, 1, 1] Assign Input2
 *                                      Output: [1, B, 1, A]
 *                                                  |
 *                                             Reshape 2D
 *                                                  |
 *                                          Transpose -> [A, B]
 *
 *
 */
class SubDecomposition : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("SubDecomposition", "0");
    SubDecomposition();
};


/**
 * @brief Convert a Multiply operation to a Group convolution with NCHW layout:
 *
 *
 *                                            Input: [A, B]
 *                                                  |
 *                                           Transpose -> [B, A]
 *                                                  |
 *                                             Reshape 4D
 *                                                  |
 * Multiply                            Group Convolution in NCHW layout
 * Input1: [A, B]            ------->    Input:  [1, B, 1, A]
 * Input2: [1, B]                       Kernel: [B, 1, 1, 1, 1] Assign Input 2
 * Output: [A, B]                       Output: [1, B, 1, A]
 *                                                  |
 *                                             Reshape 2D
 *                                                  |
 *                                          Transpose -> [A, B]
 *
 *
 */
class MulDecomposition : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("MulDecomposition", "0");
    MulDecomposition();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
