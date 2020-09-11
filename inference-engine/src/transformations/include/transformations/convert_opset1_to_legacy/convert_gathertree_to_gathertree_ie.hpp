// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

#include <ngraph/op/gather_tree.hpp>
#include <ngraph_ops/gather_tree_ie.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertGatherTreeToGatherTreeIEMatcher;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertGatherTreeToGatherTreeIEMatcher: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertGatherTreeToGatherTreeIEMatcher();
};
