// Copyright (C) 2018-2019 Intel Corporation
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

class INFERENCE_ENGINE_API_CLASS(ReshapeFullyConnectedFusion);

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ReshapeFullyConnectedFusion : public ngraph::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    ReshapeFullyConnectedFusion() : GraphRewrite() {
        construct_reshape_fc();
    }

    bool run_on_function(std::shared_ptr<ngraph::Function> f) override {
        if (!ngraph::op::util::has_op_with_type<ngraph::op::FakeQuantize>(f)) {
            return GraphRewrite::run_on_function(f);
        }
        return false;
    }

private:
    void construct_reshape_fc();
};
