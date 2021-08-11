// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API HSigmoidDecomposition;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief HSigmoidDecomposition transformation into sub-graph (min(Relu(x + 3), 6) * const(1/6).
 */
class ov::pass::HSigmoidDecomposition: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    HSigmoidDecomposition();
};
