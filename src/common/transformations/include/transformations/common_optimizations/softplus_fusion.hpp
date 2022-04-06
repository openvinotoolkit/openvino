// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API SoftPlusFusion;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief SoftPlusFusion transformation replaces group of
 * operations: log(exp(x) + 1) to SoftPlus op.
 */
class ngraph::pass::SoftPlusFusion : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("SoftPlusFusion", "0");
    SoftPlusFusion();
};
