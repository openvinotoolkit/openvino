// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <openvino/core/ov_visibility.hpp>

#include <ngraph/ngraph.hpp>

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/opsets/opset5.hpp>

using namespace std;

namespace ngraph {
namespace pass {

class OPENVINO_API BatchNormDecomposition;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::BatchNormDecomposition: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    BatchNormDecomposition();
};
