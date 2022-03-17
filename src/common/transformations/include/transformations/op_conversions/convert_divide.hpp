// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertDivide;
class TRANSFORMATIONS_API ConvertDivideWithConstant;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertDivide : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertDivide();
};

class ngraph::pass::ConvertDivideWithConstant : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertDivideWithConstant();
};
