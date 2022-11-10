// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API AddFakeQuantizeFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief AddFakeQuantizeFusion transformation replaces following graph:
 * Add->FakeQuantize to a single FakeQuantize
 * Restrictions:
 * - second input to Add is a Constant
 */
class ov::pass::AddFakeQuantizeFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("AddFakeQuantizeFusion", "0");
    AddFakeQuantizeFusion();
};

namespace ngraph {
namespace pass {
using ov::pass::AddFakeQuantizeFusion;
}  // namespace pass
}  // namespace ngraph
