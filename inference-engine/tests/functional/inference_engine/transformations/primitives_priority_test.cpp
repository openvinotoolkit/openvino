// Copyright (C) 2020 Intel Corporation
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
#include <transformations/common_optimizations/common_optimizations.hpp>
#include <transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp>
#include <transformations/convert_opset2_to_opset1/convert_opset2_to_opset1.hpp>
#include <transformations/utils/utils.hpp>
#include <convert_function_to_cnn_network.hpp>
#include <generic_ie.hpp>
#include <cpp/ie_cnn_network.h>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

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


    InferenceEngine::CNNNetwork network(f);

    // Set PrimitivesPriority to all Convolutions
    auto nGraph = network.getFunction();
    ASSERT_TRUE(nGraph);
    for (auto & op : nGraph->get_ops()) {
        if (auto conv = std::dynamic_pointer_cast<ngraph::opset1::Convolution>(op)) {
            auto & rtInfo = conv->get_rt_info();
            rtInfo["PrimitivesPriority"] = std::make_shared<ngraph::VariantWrapper<std::string> > ("test");
        }
    }


    // Force conversion from nGraph to CNNNetwork
    ngraph::pass::CommonOptimizations().run_on_function(nGraph);
    ngraph::op::GenericIE::DisableReshape noReshape(f);

    // Note: instead of running all Conversion Transformations you can make up your own transformation pipeline
    ngraph::pass::CommonOptimizations().run_on_function(nGraph);
    ngraph::pass::ConvertOpSet2ToOpSet1().run_on_function(nGraph);
    ngraph::pass::ConvertOpSet1ToLegacy().run_on_function(nGraph);
    auto clonedNetwork = InferenceEngine::details::convertFunctionToICNNNetwork(nGraph, network);

    IE_SUPPRESS_DEPRECATED_START
    InferenceEngine::CNNLayerPtr conv;
    clonedNetwork->getLayerByName("add", conv, nullptr);
    ASSERT_TRUE(conv->params.count("PrimitivesPriority"));
    ASSERT_EQ(conv->params.at("PrimitivesPriority"), "test");
    IE_SUPPRESS_DEPRECATED_END
}
