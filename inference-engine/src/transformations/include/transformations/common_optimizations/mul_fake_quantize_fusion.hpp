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

class TRANSFORMATIONS_API MulFakeQuantizeFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief MulFakeQuantizeFusion transformation replaces following graph:
 * Mul->FakeQuantize to a single FakeQuantize
 * Restrictions:
 * - second input to Mul is a Constant
 */
class ov::pass::MulFakeQuantizeFusion: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    MulFakeQuantizeFusion();
};
