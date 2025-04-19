// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief EliminateWeightlessAttributes transformation
 *
 */
class TRANSFORMATIONS_API EliminateWeightlessAttributes : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("EliminateWeightlessAttributes");
    EliminateWeightlessAttributes();
};

}  // namespace pass
}  // namespace ov
