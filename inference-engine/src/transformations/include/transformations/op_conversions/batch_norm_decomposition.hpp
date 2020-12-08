// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/ngraph.hpp>

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/opsets/opset5.hpp>

using namespace std;

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API BatchNormDecomposition;
class TRANSFORMATIONS_API BatchNormV5Decomposition;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::BatchNormDecomposition: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    BatchNormDecomposition();
};

class ngraph::pass::BatchNormV5Decomposition: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    BatchNormV5Decomposition();
};
