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

class TRANSFORMATIONS_API SoftPlusDecomposition;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief SoftPlusDecomposition transformation replaces SoftPlus op to
 * group of operations: log(exp(x) + 1).
 */
class ov::pass::SoftPlusDecomposition: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SoftPlusDecomposition();
};
