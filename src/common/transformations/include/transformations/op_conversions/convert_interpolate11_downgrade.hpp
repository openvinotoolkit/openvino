// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {
/**
 * @ingroup ie_transformation_common_api
 * @brief Converts Interpolate version 11 to Interpolate version 4 if the new op uses any of the v4 allowed
 *        interpolation modes.
 */
class TRANSFORMATIONS_API ConvertInterpolate11ToInterpolate4 : public MatcherPass {
public:
    OPENVINO_RTTI("ConvertInterpolate11ToInterpolate4", "0");
    ConvertInterpolate11ToInterpolate4();
};

}  // namespace pass
}  // namespace ov
