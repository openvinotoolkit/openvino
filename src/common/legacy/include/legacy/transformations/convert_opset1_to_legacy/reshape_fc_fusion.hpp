// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <numeric>

#include <legacy/ngraph_ops/fully_connected.hpp>
#include <ngraph/graph_util.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/pattern/op/label.hpp>
#include <ngraph/pattern/op/skip.hpp>
#include <ngraph/util.hpp>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations/utils/utils.hpp>

namespace ngraph {
namespace pass {

class ReshapeFullyConnectedFusion;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ReshapeFullyConnectedFusion : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ReshapeFullyConnectedFusion", "0");
    ReshapeFullyConnectedFusion();
};
