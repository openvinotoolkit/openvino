// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API SoftPlusFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief SoftPlusFusion transformation replaces group of
 * operations: log(exp(x) + 1) to SoftPlus op.
 */
class ov::pass::SoftPlusFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("SoftPlusFusion", "0");
    SoftPlusFusion();
};
