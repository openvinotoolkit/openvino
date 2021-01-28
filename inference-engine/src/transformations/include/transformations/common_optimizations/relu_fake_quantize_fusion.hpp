// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ReluFakeQuantizeFusion;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief ReluFakeQuantizeFusion transformation replaces following graph:
 * Relu -> FakeQuantize to FakeQuantize under following conditions:
 * -  'input_low' input to FakeQuantize is a Constant
 * -  'input_low' has non negative values
 */

class ngraph::pass::ReluFakeQuantizeFusion: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ReluFakeQuantizeFusion();
};
