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
 * @brief StridedSliceReshapeConcatFusion matches framing-like pattern built from
 * Slice/StridedSlice + Reshape + Concat and replaces it with a single Gather.
 */
class ov::pass::StridedSliceReshapeConcatFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("StridedSliceReshapeConcatFusion");
    StridedSliceReshapeConcatFusion();
};
