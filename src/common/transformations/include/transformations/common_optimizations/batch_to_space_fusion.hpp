// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API BatchToSpaceFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief BatchToSpaceFusion transformation replaces following graph:
 * Transpose (or Reshape) -> DepthToSpace -> StridedSlice -> Transpose (or Reshape)
 * to BatchToSpace
 * Restrictions:
 * - input rank must be 4
 * - Transpose permutation must be [1, 0, 2, 3]
 * - DepthToSpaceMode must be BLOCKS_FIRST
 */

class ov::pass::BatchToSpaceFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("BatchToSpaceFusion", "0");
    BatchToSpaceFusion();
};
