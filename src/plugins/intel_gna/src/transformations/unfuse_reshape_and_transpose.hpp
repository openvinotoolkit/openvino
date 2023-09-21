// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_gna {
namespace pass {

/**
 * @brief Replace 2d->4d reshape to pair of 2 reshapes (before Convolution)
 * Before:
 *      [N, HW]
 *         |
 *      Reshape
 *         |
 *   [N, C, H, W]
 *         |
 *    Convolution
 *
 * After (TransposeSinking friendly):
 *      [N, HW]
 *         |
 *      Reshape
 *         |
 *   [N, H, W, C]
 *         |
 *      Reshape
 *         |
 *   [N, C, H, W]
 *         |
 *    Convolution
 */
class Unfuse2dto4dReshapeAndTranspose : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("Unfuse2dto4dReshapeAndTranspose", "0");
    Unfuse2dto4dReshapeAndTranspose();
};

/**
 * @brief Replace 2d->4d reshape to pair of 2 reshapes (after Convolution)
 * Before:
 *    Convolution (optionally + bias/pooling/activation)
 *         |
 *    [N, C, H, W]
 *         |
 *      Reshape
 *         |
 *      [N, HW]
 *
 * After (TransposeSinking friendly):
 *    Convolution
 *         |
 *    [N, C, H, W]
 *         |
 *      Reshape
 *         |
 *   [N, H, W, C]
 *         |
 *      Reshape
 *         |
 *      [N, HW]
 *
 */
class Unfuse4dto2dReshapeAndTranspose : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("Unfuse4dto2dReshapeAndTranspose", "0");
    Unfuse4dto2dReshapeAndTranspose();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
