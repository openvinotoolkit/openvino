// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API DilatedConvolutionConverter;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief DilatedConvolutionConverter transformation replaces following graph:
 * SpaceToBatch -> Convolution -> BatchToSpace
 * to a single Convolution node with updated pads and dilations
 * Restrictions:
 * - pads in SpaceToBatch must have 0 on first and second position
 */

class ov::pass::DilatedConvolutionConverter: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    DilatedConvolutionConverter();
};
