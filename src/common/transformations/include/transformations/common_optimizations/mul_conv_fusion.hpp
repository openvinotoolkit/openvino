// Copyright (C) 2018-2022 Intel Corporation
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
class TRANSFORMATIONS_API MultiplyGroupConvolutionFusion;
class TRANSFORMATIONS_API MultiplyConvolutionBackpropDataFusion;
class TRANSFORMATIONS_API MultiplyGroupConvolutionBackpropDataFusion;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
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
 * - if the constant input to Multiply has the same rank as 'input', the constant first dimension has to be 1
 * - constant input to Multiply has to be broadcastable to weights when 'Convolution Op' is either Convolution or GroupConvolution
 * - shape of a constant input to Multiply has to be in one of following forms: (1), (1, 1, ..., 1), (C, 1, ..., 1), (1, C, 1, ..., 1)
 *   when 'Convolution Op' is either ConvolutionBackpropData or GroupConvolutionBackpropData
 */


class ngraph::pass::MultiplyConvolutionFusion: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    MultiplyConvolutionFusion();
};

class ngraph::pass::MultiplyGroupConvolutionFusion: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    MultiplyGroupConvolutionFusion();
};

class ngraph::pass::MultiplyConvolutionBackpropDataFusion: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    MultiplyConvolutionBackpropDataFusion();
};

class ngraph::pass::MultiplyGroupConvolutionBackpropDataFusion: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    MultiplyGroupConvolutionBackpropDataFusion();
};
