// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <queue>
#include <map>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/variant.hpp>
#include <transformations/utils/utils.hpp>
#include <cpp/ie_cnn_network.h>
#include <ie_ngraph_utils.hpp>
#include "transformations/rt_info/primitives_priority_attribute.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

TEST(TransformationTests, ConvBiasFusion) {
    std::shared_ptr<ngraph::Function> f(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 64, 64});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {1});
        auto bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 1, 1}, {1});
        auto conv = std::make_shared<ngraph::opset1::Convolution>(input1, weights, ngraph::Strides{1, 1}, ngraph::CoordinateDiff{0, 0},
                ngraph::CoordinateDiff{0, 0}, ngraph::Strides{1, 1});

        auto add = std::make_shared<ngraph::opset1::Add>(conv, bias);
        add->set_friendly_name("add");

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{input1});
    }

    std::unordered_map<std::string, std::string> pp;

    InferenceEngine::CNNNetwork network(f);

    // Set PrimitivesPriority to all Convolutions
    auto nGraph = network.getFunction();
    ASSERT_NE(nullptr, nGraph);
    for (auto & op : nGraph->get_ops()) {
        if (auto conv = std::dynamic_pointer_cast<ngraph::opset1::Convolution>(op)) {
            auto & rtInfo = conv->get_rt_info();
            rtInfo[ov::PrimitivesPriority::get_type_info_static()] = ov::PrimitivesPriority("test");
            pp[op->get_friendly_name()] = "test";
        }
    }

    auto clonedNetwork = InferenceEngine::details::cloneNetwork(network);
    auto funcs = clonedNetwork.getFunction();

    for (auto & op : funcs->get_ops()) {
        if (auto conv = std::dynamic_pointer_cast<ngraph::opset1::Convolution>(op)) {
            ASSERT_TRUE(pp.find(op->get_friendly_name()) != pp.end());
            ASSERT_EQ(pp[op->get_friendly_name()], "test");
        }
    }
}
