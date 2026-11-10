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
 * @brief Converts nearest-neighbour Interpolate with static shapes into a sequence of Gather operations with
 *        constant indices.
 *
 * @details Some backends lower nearest-neighbour interpolation incorrectly (or do not support it). The
 * nearest-neighbour mapping of static shapes depends only on the interpolation attributes, so it can be
 * expressed exactly with a Gather per resized axis, using indices precomputed at transformation time. This
 * pass is intended for such backends and is a no-op everywhere else.
 *
 * Only upscaling axes (scale greater than 1) are rewritten; padded, dynamic, antialiased and downscaling
 * cases, as well as non-nearest interpolation modes, are left untouched.
 */
class TRANSFORMATIONS_API ConvertNearestInterpolateToGather : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertNearestInterpolateToGather");
    ConvertNearestInterpolateToGather();
};

}  // namespace pass
}  // namespace ov
