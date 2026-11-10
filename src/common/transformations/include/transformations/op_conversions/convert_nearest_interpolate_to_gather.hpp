// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {
/**
 * @ingroup ov_transformation_common_api
 * @brief Replaces nearest-neighbour Interpolate with a sequence of Gather operations using constant indices.
 *
 * Some backends lower nearest-neighbour interpolation incorrectly, or do not support it. The nearest-neighbour
 * mapping of a statically-shaped Interpolate depends only on its attributes and the input/output shapes, so it
 * can be expressed exactly with one Gather per resized axis whose indices are computed at transformation time.
 *
 * Before:                                     After:
 *
 *   Data                                        Data
 *     |                                           |
 *   [Interpolate]  mode = nearest               [Gather]  axis = 2 (height indices)
 *     |            shape_calculation_mode      /        \
 *   Result         = scales or sizes          |          |
 *                                             [Gather]  axis = 3 (width indices)
 *                                                |
 *                                              Result
 *
 * The rewrite is applied only when all of the following hold:
 *  - the op is Interpolate-4 or Interpolate-11 with mode = nearest, antialias = false and no padding;
 *  - the input and output shapes are static, so the index mapping is known at transformation time;
 *  - the scales/sizes input (and the axes input, when present) is constant-foldable;
 *  - at least one interpolated axis is upscaling (scale greater than 1). Downscaling axes and linear/cubic
 *    modes are left untouched: the decomposition is exact but only upscaling is affected by the reported
 *    backend defect, so the pass stays out of the way elsewhere.
 */
class TRANSFORMATIONS_API ConvertNearestInterpolateToGather : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertNearestInterpolateToGather");
    ConvertNearestInterpolateToGather();
};

}  // namespace pass
}  // namespace ov
