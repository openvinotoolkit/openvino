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
 * @brief Replaces LINEAR_ONNX Interpolate with weighted Gather operations using constant indices and weights.
 *
 * Some backends lower linear interpolation correctly but convert its output between layouts incorrectly. The linear
 * (LINEAR_ONNX) mapping of a statically-shaped Interpolate depends only on its attributes and the input/output
 * shapes: every output sample is a two-tap blend of the clamped neighbouring input samples (the reference
 * implementation uses weight d2 on the lower index and d1 on the upper one). It can therefore be expressed exactly
 * with two Gathers, two Multiply and one Add per resized axis, using weights and indices precomputed at
 * transformation time.
 *
 * Before:                                     After (per resized axis):
 *
 *   Data                                        Data
 *     |                                           |
 *   [Interpolate]  mode = linear_onnx           [Gather]  lower index -----------\
 *     |            any coordinate mode          [Gather]  upper index ---------+->[Multiply] x d2 / x d1
 *   Result                                          \_________________________/            |
 *                                                                                    [Add] -> next axis or Result
 *
 * The rewrite is applied only when all of the following hold:
 *  - the op is Interpolate-4 or Interpolate-11 with mode = linear_onnx, antialias = false and no padding;
 *  - the input and output shapes are static, so the mapping is known at transformation time;
 *  - the scales/sizes input (and the axes input, when present) is constant-foldable;
 *  - the element type is f32, f16 or bf16 (other types are computed in f32 internally, matching the reference).
 */
class TRANSFORMATIONS_API ConvertLinearInterpolateToGather : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertLinearInterpolateToGather");
    ConvertLinearInterpolateToGather();
};

}  // namespace pass
}  // namespace ov
