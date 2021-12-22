// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <openvino/core/ov_visibility.hpp>

namespace ngraph {
namespace pass {

class OPENVINO_API DivideFusion;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief DivideFusion transformation replaces a sub-graph
 * Pow(y, -1) * x or x * Pow(y, -1) with Divide(x,y)
 */
class ngraph::pass::DivideFusion : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    DivideFusion();
};
