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

class TRANSFORMATIONS_API AdaptivePoolToReduce;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief AdaptivePoolToReduce transformation replaces AdaptiveXXXPool with ReduceXXX when possible
 */

class ov::pass::AdaptivePoolToReduce : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("AdaptivePoolToReduce");
    AdaptivePoolToReduce();
};
