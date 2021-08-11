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

class TRANSFORMATIONS_API TransposeToReshape;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief TransposeToReshape transformation replaces suitable Transposes with Reshape operation or optimizes them out
 */
class ov::pass::TransposeToReshape : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    TransposeToReshape();
};
