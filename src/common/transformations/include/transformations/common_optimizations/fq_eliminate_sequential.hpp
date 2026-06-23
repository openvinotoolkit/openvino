// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API FakeQuantizeEliminateSequential;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief FakeQuantizeEliminateSequential folds two sequential FakeQuantize operations (FQ1 -> FQ2)
 * into a single one. The notation below is FQ(in_low, in_high, out_low, out_high, levels).
 *
 * FQ1 and FQ2 may be separated by one or several value-preserving Reshape/Transpose/Squeeze/Unsqueeze
 * ops. A matched FakeQuantize is per-tensor (scalar ranges) and therefore applied element-wise, so it
 * commutes with such ops and the folding stays valid through the chain.
 *
 * The transformation applies only when:
 *  - all four range bounds of both ops are finite constants;
 *  - FQ1 output range lies within FQ2 input range, otherwise FQ2 clamps FQ1 output;
 *  - both ops have more than one level;
 *  - FQ1 output grid aligns with FQ2 output grid: the offset between the grids and the FQ1 step are
 *    integer multiples of the FQ2 step, so re-quantization preserves every FQ1 level.
 *
 * When these conditions hold:
 *  - Elimination: if FQ2 is identity on its range (input range equals output range), it only
 *    re-quantizes onto an aligned super-grid and is removed, keeping FQ1. For example:
 *      FQ1(-1, 1, -1, 1, 256) -> FQ2(-2, 2, -2, 2, 1021)  =>  FQ1(-1, 1, -1, 1, 256)
 *  - Merge: otherwise, if FQ1 and FQ2 have the same level count, they are replaced by a single
 *    FakeQuantize with FQ1's input range and FQ2's output range. For example:
 *      FQ1(-2, 2, -1, 1, 256) -> FQ2(-1, 1, -1, 0, 256)   =>  FQ(-2, 2, -1, 0, 256)
 */
class ov::pass::FakeQuantizeEliminateSequential : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FakeQuantizeEliminateSequential");
    FakeQuantizeEliminateSequential();
};
