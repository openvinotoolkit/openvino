// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_cpu {

/**
 * @brief Decompose 1D ConvolutionBackpropData (3D tensors) into ACL-optimized 2D operations
 *
 * Before:
 *
 *    +----------------+    +-------------------+
 *    | Input[N,C,L]   |    | Weights[C,C',K]   |
 *    +--------+-------+    +---------+---------+
 *             |                      |
 *        +----v----------------------v-----+
 *        | ConvolutionBackpropData1D       |
 *        | (stride, pad, dilation)         |
 *        +----------------+----------------+
 *                         |
 *                +--------v----------+
 *                | Output[N,C',L']   |
 *                +-------------------+
 *
 * After:
 *
 *    +----------------+    +-------------------+
 *    | Input[N,C,L]   |    | Weights[C,C',K]   |
 *    +--------+-------+    +---------+---------+
 *             |                      |
 *      +------v--------+      +------v--------+
 *      | Reshape to 4D |      | Reshape to 4D |
 *      | [N,C,L,1]     |      | [C,C',K,1]    |
 *      +------+--------+      +------+--------+
 *             |                      |
 *             |               +------v--------+
 *             |               | Transpose     |
 *             |               | [C',C,K,1]    |
 *             |               +------+--------+
 *             |                      |
 *      +------v--------+             |
 *      | ScatterUpdate |             |
 *      | (if stride>1) |             |
 *      +------+--------+             |
 *             |                      |
 *      +------v--------+             |
 *      | Pad           |             |
 *      | (if needed)   |             |
 *      +------+--------+             |
 *             |                      |
 *        +----v----------------------v-----+
 *        | Convolution2D                   |
 *        | (stride=1, pad=0)               |
 *        +----------------+----------------+
 *                         |
 *                  +------v--------+
 *                  | Reshape to 3D |
 *                  | [N,C',L']     |
 *                  +------+--------+
 *                         |
 *                +--------v----------+
 *                | Output[N,C',L']   |
 *                +-------------------+
 *
 * This transformation enables ACL optimization for 1D deconvolution by converting
 * it into supported 2D operations. The upsampling is done via ScatterUpdate to
 * insert zeros between input elements when stride > 1.
 */
class Deconv1DDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("Deconv1DDecomposition");
    Deconv1DDecomposition();
};

}  // namespace ov::intel_cpu
