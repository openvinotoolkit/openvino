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

class TRANSFORMATIONS_API MishFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief MishFusion transformation replaces group of
 * operations: x * tanh(log(exp(x) + 1)) to Mish op.
 */
class ov::pass::MishFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MishFusion", "0");
    MishFusion();
};

namespace ngraph {
namespace pass {
using ov::pass::MishFusion;
}  // namespace pass
}  // namespace ngraph
