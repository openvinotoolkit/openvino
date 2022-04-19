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

class TRANSFORMATIONS_API TransposeToReshape;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief TransposeToReshape transformation replaces suitable Transposes with Reshape operation or optimizes them out
 */
class ngraph::pass::TransposeToReshape : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("TransposeToReshape", "0");
    TransposeToReshape();
};
