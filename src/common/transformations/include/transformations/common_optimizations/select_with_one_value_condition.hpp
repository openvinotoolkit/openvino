// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API SelectWithOneValueCondition;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief SelectWithOneValueCondition transformation eliminates Select operation if the condition
 * is constant and consists of al True or False elements
 */

class ov::pass::SelectWithOneValueCondition : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("SelectWithOneValueCondition", "0");
    SelectWithOneValueCondition();
};
