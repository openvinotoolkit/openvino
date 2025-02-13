// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ReshapeSequenceFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief ReshapeSequenceFusion fuses sequence of Reshape operation into single Reshape or eliminates full redundant
 * sequence
 */

class ov::pass::ReshapeSequenceFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ReshapeSequenceFusion");
    ReshapeSequenceFusion(bool use_shape_for_elimination = true);
};
