// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

#include "ngraph/pattern/matcher.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API SoftPlusToMishFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief SoftPlusToMishFusion transformation replaces group of
 * operations: x * tanh(softplus(x)) to Mish op.
 */
class ov::pass::SoftPlusToMishFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("SoftPlusToMishFusion", "0");
    SoftPlusToMishFusion();
};

namespace ngraph {
namespace pass {
using ov::pass::SoftPlusToMishFusion;
}  // namespace pass
}  // namespace ngraph
