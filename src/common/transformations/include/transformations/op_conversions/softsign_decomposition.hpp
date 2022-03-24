// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API SoftSignDecomposition;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief SoftSignDecomposition transformation replaces softmax with the graph, that matches a formula:
 * SoftSign(x) = x / (1 + |x|)
 */

class ngraph::pass::SoftSignDecomposition : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("SoftSignDecomposition", "0");
    SoftSignDecomposition();
};
