// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

using namespace std;

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API BatchNormDecomposition;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::BatchNormDecomposition : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("BatchNormDecomposition", "0");
    BatchNormDecomposition();
};
