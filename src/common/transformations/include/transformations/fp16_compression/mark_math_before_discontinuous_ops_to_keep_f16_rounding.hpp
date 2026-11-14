// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief Marks Math-type operations that directly feed a discontinuous operation with
 * disable_conversion(f16, f32).
 *
 * Plugins that only execute Math ops (Cos, Sin, ...) in f32 normally let ConvertPrecision widen
 * f16 edges to f32, which drops the f16 rounding step. That rounding matters right before an operation
 * whose result jumps at some value: e.g. cos(x) == 0.99998 in f32 stays < 1.0, but rounds up to exactly
 * 1.0 in f16, changing the Floor result. The discontinuous operations handled are Floor, Ceiling, Round,
 * Sign, Mod and FloorMod (Mod and FloorMod are matched on either of their inputs).
 * This pass only marks the Math node when it is immediately followed by one of them, so the fix is scoped
 * to the confirmed-problematic pattern instead of every Math op in the model.
 *
 * +-----------+     +---------------------------------------------+     +-------+
 * | Math (f16)|---->| Floor / Ceiling / Round / Sign / Mod / FloorMod |---->| ...   |
 * +-----------+     +---------------------------------------------+     +-------+
 *
 * The actual f16 I/O preservation (re-narrowing inputs, widening the output for f32 consumers) is
 * performed later by the plugin-specific ConvertPrecision type_to_fuse callback, which checks this
 * rt_info marker before acting.
 */
class TRANSFORMATIONS_API MarkMathBeforeDiscontinuousOpsToKeepF16Rounding : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MarkMathBeforeDiscontinuousOpsToKeepF16Rounding");
    MarkMathBeforeDiscontinuousOpsToKeepF16Rounding();
};

}  // namespace ov::pass
