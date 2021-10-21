// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API RemoveLoopDanglingParameters;

}  // namespace pass
}  // namespace ngraph

/*
 * @ingroup ie_transformation_common_api
 * @brief RemoveLoopDanglingParameters transformation
 * removed Loop inputs which are not connected to other nodes
 * in the body of a Loop
 */

class ngraph::pass::RemoveLoopDanglingParameters: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    RemoveLoopDanglingParameters();
};
