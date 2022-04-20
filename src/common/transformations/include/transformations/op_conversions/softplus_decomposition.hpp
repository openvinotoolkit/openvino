// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API SoftPlusDecomposition;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief SoftPlusDecomposition transformation replaces SoftPlus op to
 * group of operations: log(exp(x) + 1).
 */
class ngraph::pass::SoftPlusDecomposition : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("SoftPlusDecomposition", "0");
    SoftPlusDecomposition();
};
