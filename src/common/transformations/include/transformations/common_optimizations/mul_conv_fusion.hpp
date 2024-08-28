// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MultiplyConvolutionFusion;
class TRANSFORMATIONS_API MultiplyGroupConvolutionFusion;
class TRANSFORMATIONS_API MultiplyConvolutionBackpropDataFusion;
class TRANSFORMATIONS_API MultiplyGroupConvolutionBackpropDataFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief Multiply->Convolution fusion replaces following graph:
 *
 *   +-------+   +----------+
 *   | Input |   | Constant |
 *   +-------+   +----------+
 *       |            |
 *       ------  ------
 *            |  |
 *            v  v
 *         +----------+            +---------+
 *         | Multiply |            | Weights |
 *         +----------+            +---------+
 *              |                       |
 *              -----------    ----------
 *                        |    |
 *                        v    v
 *                   +----------------+
 *                   | Convolution Op |
 *                   +----------------+
 *
 * to following:
 *
 *                           +---------+   +----------+
 *                           | Weights |   | Constant |
 *                           +---------+   +----------+
 *                                |            |
 *                                ------  ------
 *                                     |  |
 *                                     v  v
 *          +-------+              +----------+
 *          | Input |              | Multiply |
 *          +-------+              +----------+
 *              |                       |
 *              -----------    ----------
 *                        |    |
 *                        v    v
 *                   +----------------+
 *                   | Convolution Op |
 *                   +----------------+
 *
 * where 'Convolution Op' is one of:
 * - Convolution
 * - ConvolutionBackpropData
 * - GroupConvolution
 * - GroupConvolutionBackpropData
 *
 * Restrictions:
 * - weights' shape is static
 * - if the constant input to Multiply has the same rank as weights, the constant first dimension has to be 1
 * - constant input to Multiply has to be broadcastable to weights when 'Convolution Op' is either Convolution or
 * GroupConvolution
 * - shape of a constant input to Multiply has to be in one of following forms: (1), (1, 1, ..., 1), (C, 1, ..., 1), (1,
 * C, 1, ..., 1) when 'Convolution Op' is either ConvolutionBackpropData or GroupConvolutionBackpropData
 */

class ov::pass::MultiplyConvolutionFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MultiplyConvolutionFusion", "0");
    MultiplyConvolutionFusion();
};

class ov::pass::MultiplyGroupConvolutionFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MultiplyGroupConvolutionFusion", "0");
    MultiplyGroupConvolutionFusion();
};

class ov::pass::MultiplyConvolutionBackpropDataFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MultiplyConvolutionBackpropDataFusion", "0");
    MultiplyConvolutionBackpropDataFusion();
};

class ov::pass::MultiplyGroupConvolutionBackpropDataFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MultiplyGroupConvolutionBackpropDataFusion", "0");
    MultiplyGroupConvolutionBackpropDataFusion();
};
