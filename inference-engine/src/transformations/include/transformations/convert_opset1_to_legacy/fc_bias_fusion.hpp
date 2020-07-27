// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph_ops/fully_connected.hpp>
#include <ngraph/builder/make_constant.hpp>
#include <ngraph/graph_util.hpp>
#include <ngraph/op/add.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/pattern/op/label.hpp>
#include <ngraph/pattern/op/skip.hpp>
#include <ngraph/util.hpp>
#include <ngraph/ngraph.hpp>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/rt_info.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API FullyConnectedBiasFusion;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::FullyConnectedBiasFusion : public ngraph::pass::MatcherPass {
public:
    FullyConnectedBiasFusion();
};
