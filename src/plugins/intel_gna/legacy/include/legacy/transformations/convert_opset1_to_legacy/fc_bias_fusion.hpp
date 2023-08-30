// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_api.h>

#include <functional>
#include <legacy/ngraph_ops/fully_connected.hpp>
#include <memory>
#include <ngraph/graph_util.hpp>
#include <ngraph/ngraph.hpp>
#include <ngraph/op/add.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/pattern/op/label.hpp>
#include <ngraph/pattern/op/skip.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/util.hpp>

namespace ngraph {
namespace pass {

class FullyConnectedBiasFusion;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::FullyConnectedBiasFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("FullyConnectedBiasFusion", "0");
    FullyConnectedBiasFusion();
};
