// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertPadToGroupConvolution;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief ConvertPadToGroupConvolution transformation replaces Pad operation
 * with GroupConvolution but has some restrictions on Pad parameters:
 * 1. PadMode must be Constant and value is equal to 0
 * 2. Padding must be applied only for spatial dimensions
 * 3. Input shape rank must be static and greater than 3
 * 4. Padding values must be non-negative
 */

class ov::pass::ConvertPadToGroupConvolution : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertPadToGroupConvolution");
    ConvertPadToGroupConvolution();
};
