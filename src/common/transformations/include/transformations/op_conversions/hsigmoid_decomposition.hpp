// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API HSigmoidDecomposition;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief HSigmoidDecomposition transformation into sub-graph (min(Relu(x + 3), 6) * const(1/6).
 */
class ngraph::pass::HSigmoidDecomposition : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("HSigmoidDecomposition", "0");
    HSigmoidDecomposition();
};
