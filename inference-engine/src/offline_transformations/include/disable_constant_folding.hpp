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

class TRANSFORMATIONS_API DisablePriorBoxConstantFolding;
class TRANSFORMATIONS_API DisablePriorBoxClusteredConstantFolding;
class TRANSFORMATIONS_API DisableShapeOfConstantFolding;
class TRANSFORMATIONS_API EnableShapeOfConstantFolding;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::DisableShapeOfConstantFolding : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    DisableShapeOfConstantFolding();
};

class ngraph::pass::EnableShapeOfConstantFolding : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    EnableShapeOfConstantFolding();
};

class ngraph::pass::DisablePriorBoxConstantFolding : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    DisablePriorBoxConstantFolding();
};

class ngraph::pass::DisablePriorBoxClusteredConstantFolding : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    DisablePriorBoxClusteredConstantFolding();
};