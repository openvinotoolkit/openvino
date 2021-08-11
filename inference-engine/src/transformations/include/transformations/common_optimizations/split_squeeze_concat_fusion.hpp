// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/ngraph.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "ngraph/pattern/matcher.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API SplitSqueezeConcatFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief SplitSqueezeConcatFusion transformation replaces group of
 * operations: Split -> Squeeze (multiple) -> Concat to Transpose -> Reshape ops.
 */
class ov::pass::SplitSqueezeConcatFusion : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SplitSqueezeConcatFusion();
};
