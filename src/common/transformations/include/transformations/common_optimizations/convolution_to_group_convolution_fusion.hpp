// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief ConvolutionToGroupConvolutionFusion transformation replaces following graph:
 *                    Split (or VariadicSplit)
 *                  /       \
 *                Conv ... Conv
 *                  \       /
 *                   \     /
 *                    Concat
 *
 * to GroupConvolution
 */
class TRANSFORMATIONS_API ConvolutionToGroupConvolutionFusion : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvolutionToGroupConvolutionFusion");
    ConvolutionToGroupConvolutionFusion();
};

}  // namespace pass
}  // namespace ov
