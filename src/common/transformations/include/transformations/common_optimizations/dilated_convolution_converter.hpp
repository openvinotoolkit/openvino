// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API DilatedConvolutionConverter;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief DilatedConvolutionConverter transformation replaces following graph:
 * SpaceToBatch -> Convolution -> BatchToSpace
 * to a single Convolution node with updated pads and dilations
 * Restrictions:
 * - pads in SpaceToBatch must have 0 on first and second position
 */

class ngraph::pass::DilatedConvolutionConverter : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("DilatedConvolutionConverter", "0");
    DilatedConvolutionConverter();
};
