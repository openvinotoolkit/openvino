// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_gna {
namespace pass {

/**
 * @brief Convert a padded convolution with bias, max pooling and activation function
 * wrapped with transposes, to a valid convolution with padding added before the leading transpose,
 * POT precessed models are supported (fake quantized layers omitted below for clarity):
 *
 *                                                  Padding
 *                                                     |
 *   Transpose (NHWC -> NCHW)               Transpose (NHWC -> NCHW)
 *              |                                      |
 *   Convolution with padding                  Valid convolution
 *              |                                      |
 *   Broadcast Bias (optional)              Broadcast Bias (optional)
 *              |                                      |
 *    Max Pooling (optional)                 Max Pooling (optional)
 *              |                                      |
 * Activation Function (optional)       Activation Function (optional)
 *              |                                      |
 *   Transpose (NCHW -> NHWC)               Transpose (NCHW -> NHWC)
 *
 */
class ConvertPaddedToValidConv : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertPaddedToValidConv", "0");
    ConvertPaddedToValidConv();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
