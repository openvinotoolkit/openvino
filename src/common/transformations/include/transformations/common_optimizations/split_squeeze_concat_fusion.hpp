// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API SplitSqueezeConcatFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief SplitSqueezeConcatFusion transformation replaces group of
 * operations: Split -> Squeeze (multiple) -> Concat to Transpose -> Reshape ops.
 */
class ov::pass::SplitSqueezeConcatFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("SplitSqueezeConcatFusion");
    SplitSqueezeConcatFusion(bool use_shapes);
};
