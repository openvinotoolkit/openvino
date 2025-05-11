// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API RandomUniformFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief RandomUniformFusion transformation replaces RandomUniform -> Add or
 * RandomUniform -> Mul subgraph with a RandomUniform and replaces min and max const
 * with corrected values.
 */
class ov::pass::RandomUniformFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RandomUniformFusion");
    RandomUniformFusion();
};
