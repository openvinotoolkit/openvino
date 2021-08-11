// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

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
class ov::pass::AddFakeQuantizeFusion: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    AddFakeQuantizeFusion();
};
