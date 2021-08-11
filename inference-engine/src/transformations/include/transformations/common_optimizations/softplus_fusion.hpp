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

class TRANSFORMATIONS_API SoftPlusFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief SoftPlusFusion transformation replaces group of
 * operations: log(exp(x) + 1) to SoftPlus op.
 */
class ov::pass::SoftPlusFusion: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SoftPlusFusion();
};
