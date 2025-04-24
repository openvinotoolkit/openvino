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

class TRANSFORMATIONS_API SelectWithOneValueCondition;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief SelectWithOneValueCondition transformation eliminates Select operation if the condition
 * is constant and consists of al True or False elements
 */

class ov::pass::SelectWithOneValueCondition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("SelectWithOneValueCondition");
    SelectWithOneValueCondition();
};
