// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <functional>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API MultiplyConvolutionFusion;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief MultiplyConvolutionFusion transformation replaces following graph:
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
 * - constant input to Multiply has to be broadcastable to weights
 */
class ngraph::pass::MultiplyConvolutionFusion: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    MultiplyConvolutionFusion();
};
