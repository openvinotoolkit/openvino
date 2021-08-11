// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertPadToGroupConvolution;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertPadToGroupConvolution transformation replaces Pad operation
 * with GroupConvolution but has some restrictions on Pad parameters:
 * 1. PadMode must be Constant and value is equal to 0
 * 2. Padding must be applied only for spatial dimensions
 * 3. Input shape rank must be static and greater than 3
 */

class ov::pass::ConvertPadToGroupConvolution: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertPadToGroupConvolution();
};
