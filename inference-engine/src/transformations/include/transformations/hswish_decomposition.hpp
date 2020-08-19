// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API HSwishDecomposition;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief HSwishDecomposition transformation into sub-graph x * (min(Relu(x + 3), 6) * const(1/6).
 */
class ngraph::pass::HSwishDecomposition: public ngraph::pass::MatcherPass {
public:
    HSwishDecomposition();
};
