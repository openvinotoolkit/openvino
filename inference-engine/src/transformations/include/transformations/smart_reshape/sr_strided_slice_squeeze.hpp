// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API StridedSliceSqueeze;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief StridedSliceSqueeze transformation looks for SS -> Squeeze and corrects SS inputs and attributes for SS output to be squeeze-able
 */

class ngraph::pass::StridedSliceSqueeze : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    StridedSliceSqueeze();
};
