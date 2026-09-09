// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API StridedSliceReshapeConcatFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief Detects and fuses framing-like subgraphs made of Slice/StridedSlice + Reshape + Concat.
 *
 * This matcher pass identifies branches that extract windows from the same 2D input tensor,
 * reshapes each window from [B, W] to [B, 1, W], and concatenates them along axis=1.
 * The pass replaces the whole pattern with a single Gather.
 *
 * ## Before (for illustration purpose)
 *
 *                    Input [B, N]
 *                         |
 *         +---------------+---------------+
 *         |               |               |
 *  StridedSlice/Slice  StridedSlice/Slice  ...
 *      [B, W]             [B, W]
 *         |               |
 *    Reshape [B,1,W]  Reshape [B,1,W]
 *         |               |
 *         +------- Concat(axis=1) -------+
 *                         |
 *                    Output [B, K, W]
 *
 * ## After
 *
 *                    Input [B, N]
 *                         |
 *             Gather(axis=1, indices[K,W])
 *                         |
 *                    Output [B, K, W]
 *
 * ## Notes
 * - All branches must read from the same source tensor.
 * - Branch window sizes must be equal (same W).
 * - This pass covers both Slice and StridedSlice forms under supported constraints.
 */
class ov::pass::StridedSliceReshapeConcatFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("StridedSliceReshapeConcatFusion");
    StridedSliceReshapeConcatFusion();
};
