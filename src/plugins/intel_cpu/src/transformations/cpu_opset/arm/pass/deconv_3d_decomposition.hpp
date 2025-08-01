// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_cpu {

/**
 * @brief Decompose 3D ConvolutionBackpropData into ACL-optimized 2D operations
 *
 * Before:
 *
 *    +----------------+    +-------------------+
 *    | Input[N,C,L]   |    | Weights[C,C',K]   |
 *    +--------+-------+    +---------+---------+
 *             |                      |
 *        +----v----------------------v-----+
 *        | ConvolutionBackpropData3D       |
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
 * This transformation enables ACL optimization for 3D deconvolution by converting
 * it into supported 2D operations. The upsampling is done via ScatterUpdate to
 * insert zeros between input elements when stride > 1.
 */
class Deconv3DDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("Deconv3DDecomposition");
    Deconv3DDecomposition();
};

}  // namespace ov::intel_cpu
