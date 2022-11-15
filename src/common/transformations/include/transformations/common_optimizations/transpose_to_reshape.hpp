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

class TRANSFORMATIONS_API TransposeToReshape;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief TransposeToReshape transformation replaces suitable Transposes with Reshape operation or optimizes them out
 */
class ov::pass::TransposeToReshape : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("TransposeToReshape", "0");
    TransposeToReshape();
};

namespace ngraph {
namespace pass {
using ov::pass::TransposeToReshape;
}  // namespace pass
}  // namespace ngraph
