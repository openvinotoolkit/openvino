// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

#include "ngraph/pattern/matcher.hpp"

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API SplitSqueezeConcatFusion;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief SplitSqueezeConcatFusion transformation replaces group of
 * operations: Split -> Squeeze (multiple) -> Concat to Transpose -> Reshape ops.
 */
class ngraph::pass::SplitSqueezeConcatFusion : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("SplitSqueezeConcatFusion", "0");
    SplitSqueezeConcatFusion();
};
