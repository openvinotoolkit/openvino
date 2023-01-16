// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/pass/graph_rewrite.hpp>
#include <openvino/pass/pattern/matcher.hpp>
#include <transformations_visibility.hpp>
#include <vector>

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
