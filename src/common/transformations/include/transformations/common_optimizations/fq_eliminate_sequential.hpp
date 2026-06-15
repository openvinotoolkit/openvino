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
 * @brief FakeQuantizeEliminateSequential optimizes sequential FakeQuantize operations by eliminating or merging them
 * when: Elimination (FQ2 is removed):
 *  - Case 1: all parameters and levels match exactly.
 *  - Case 2-5: when FQ2 is identity on its range and grids align:
 *    * Case 2: second FakeQuantize is identity on its own range (input range equals output range).
 *    * Case 3: the first output range is a subrange of the second output range.
 *    * Case 4: the second FakeQuantize has a degenerate grid (zero step) and ranges match exactly.
 *    * Case 5: the first output grid is aligned to the second output grid (integer multiple offset and step).
 *
 *  Merging (both FQ1 and FQ2 are replaced by a single merged FQ):
 *  - Case 6: when grids align, FQ2 is not identity on its range, and both have the same level count.
 *    Creates a merged FQ with FQ1's input range and FQ2's output range.
 *
 * Examples:
 *  - Case 1 (exact match):
 *    FQ1(x, [-1, 1] -> [-1, 1], levels=256) -> FQ2(x, [-1, 1] -> [-1, 1], levels=256) => FQ1
 *  - Case 2 (identity on own range):
 *    FQ1(x, [-1, 1] -> [-1, 1], levels=256) -> FQ2(x, [-2, 2] -> [-2, 2], levels=511) => FQ1
 *  - Case 5 (aligned grids):
 *    FQ1(x, [-1, 1] -> [-1, 1], levels=256) -> FQ2(x, [-2, 2] -> [-2, 2], levels=511) => FQ1
 *  - Case 6 (merge with non-identity FQ2):
 *    FQ1(x, [-2, 2] -> [-1, 1], levels=256) -> FQ2(x, [-1, 1] -> [-0.5, 0.5], levels=256) =>
 *    Merged FQ(x, [-2, 2] -> [-0.5, 0.5], levels=256)
 */
class ov::pass::FakeQuantizeEliminateSequential : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FakeQuantizeEliminateSequential");
    FakeQuantizeEliminateSequential();
};
