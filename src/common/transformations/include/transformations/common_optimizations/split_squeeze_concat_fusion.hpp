// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

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
    OPENVINO_RTTI("SplitSqueezeConcatFusion", "0");
    SplitSqueezeConcatFusion();
};

namespace ngraph {
namespace pass {
using ov::pass::SplitSqueezeConcatFusion;
}  // namespace pass
}  // namespace ngraph
