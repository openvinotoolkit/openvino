// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

#include "ngraph/pattern/matcher.hpp"

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API MishFusion;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief MishFusion transformation replaces group of
 * operations: x * tanh(log(exp(x) + 1)) to Mish op.
 */
class ngraph::pass::MishFusion : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("MishFusion", "0");
    MishFusion();
};
