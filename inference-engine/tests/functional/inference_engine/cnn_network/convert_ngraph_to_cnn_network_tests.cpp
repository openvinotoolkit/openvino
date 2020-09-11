// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cpp/ie_cnn_network.h>
#include <legacy/cnn_network_impl.hpp>  // deprecated API

#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/common_optimizations.hpp>
#include <transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp>
#include <transformations/convert_opset2_to_opset1/convert_opset2_to_opset1.hpp>
#include <transformations/convert_opset3_to_opset2/convert_opset3_to_opset2.hpp>
#include <transformations/convert_precision.hpp>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph_ops/convolution_ie.hpp>
#include <transformations/init_node_info.hpp>
#include <legacy/convert_function_to_cnn_network.hpp>

using namespace testing;
using namespace InferenceEngine;

TEST(ConvertFunctionToCNNNetworkTests, ConvertPReLUNetwork) {
    std::shared_ptr<ngraph::Function> f;
    {
        auto param1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2, 2});
        auto param2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2, 2});
        auto prelu = std::make_shared<ngraph::opset1::PRelu>(param1, param2);
        prelu->set_friendly_name("prelu");
        auto result = std::make_shared<ngraph::op::Result>(prelu);

        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                               ngraph::ParameterVector{param1, param2});
        ngraph::pass::InitNodeInfo().run_on_function(f);
    }

    InferenceEngine::CNNNetwork nGraphImpl(f);
    try {
        auto net = std::make_shared<InferenceEngine::details::CNNNetworkImpl>(
            static_cast<const InferenceEngine::ICNNNetwork &>(nGraphImpl));
    } catch (InferenceEngine::details::InferenceEngineException &err) {
        const std::string ref_msg = "Error of validate layer: prelu with type: PReLU. Number of inputs (2) is not equal to expected ones: 1";
        const std::string resp_msg = err.what();
        ASSERT_TRUE(resp_msg.find(ref_msg) != std::string::npos) << resp_msg;
    }
}

TEST(ConvertFunctionToCNNNetworkTests, ConvertConvolutionNetwork) {
    std::shared_ptr<ngraph::Function> f;
    {
        auto param1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 64, 64});
        auto param2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1});
        auto convolution = std::make_shared<ngraph::op::ConvolutionIE>(param1, param2,
                                                                  ngraph::Strides{1, 1},
                                                                  ngraph::Strides{1, 1},
                                                                  ngraph::CoordinateDiff{0, 0},
                                                                  ngraph::CoordinateDiff{0, 0});
        convolution->set_friendly_name("convolution");
        auto result = std::make_shared<ngraph::op::Result>(convolution);

        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                               ngraph::ParameterVector{param1, param2});
        ngraph::pass::InitNodeInfo().run_on_function(f);
    }

    InferenceEngine::CNNNetwork nGraphImpl(f);
    try {
        auto net = std::make_shared<InferenceEngine::details::CNNNetworkImpl>(
            static_cast<const InferenceEngine::ICNNNetwork &>(nGraphImpl));
    } catch (InferenceEngine::details::InferenceEngineException &err) {
        FAIL();
    }
}

TEST(ConvertFunctionToCNNNetworkTests, OpsShouldBeConvertedToIERepresentation) {
    ngraph::NodeVector should_converted_to_ie = {
            std::make_shared<ngraph::opset4::Broadcast>(),
            std::make_shared<ngraph::opset4::Convolution>(),
            std::make_shared<ngraph::opset4::ConvolutionBackpropData>(),
            std::make_shared<ngraph::opset4::Gather>(),
            std::make_shared<ngraph::opset4::GatherTree>(),
            std::make_shared<ngraph::opset4::GroupConvolution>(),
            std::make_shared<ngraph::opset4::GroupConvolutionBackpropData>(),
            std::make_shared<ngraph::opset4::GRUCell>(),
            // std::make_shared<ngraph::op::v5::GRUSequence>(), todo: enable after GRUSequence support
            std::make_shared<ngraph::opset4::HardSigmoid>(),
            std::make_shared<ngraph::opset4::LRN>(),
            std::make_shared<ngraph::opset4::LSTMCell>(),
            // std::make_shared<ngraph::op::v5::LSTMSequence>(), todo: enable after LSTMSequence support
            std::make_shared<ngraph::opset4::NonMaxSuppression>(),
            std::make_shared<ngraph::opset4::NormalizeL2>(),
            std::make_shared<ngraph::opset4::RNNCell>(),
            // std::make_shared<ngraph::op::v5::RNNSequence>(), todo: enable after RNNSequence support
            std::make_shared<ngraph::opset4::OneHot>(),
            std::make_shared<ngraph::opset4::Pad>(),
            std::make_shared<ngraph::opset4::PriorBoxClustered>(),
            std::make_shared<ngraph::opset4::PriorBox>(),
            std::make_shared<ngraph::opset4::Proposal>(),
            std::make_shared<ngraph::opset4::Selu>(),
            std::make_shared<ngraph::opset4::Swish>(),
            std::make_shared<ngraph::opset4::Tile>(),
            std::make_shared<ngraph::opset4::TopK>(),
    };

    // create simple ngraph function Parameter -> Result
    std::shared_ptr<ngraph::Function> f;
    auto param = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{});
    auto res = std::make_shared<ngraph::opset4::Result>(param);
    f = std::make_shared<ngraph::Function>(ngraph::ResultVector{res}, ngraph::ParameterVector{param});
    InferenceEngine::CNNNetwork nGraphImpl(f);

    for (const auto& ngraph_node : should_converted_to_ie) {
        // add node without inputs to the ngraph function
        ngraph_node->set_output_type(0, ngraph::element::f32, ngraph::Shape{});
        res->input(0).replace_source_output(ngraph_node->output(0));

        EXPECT_THROW(InferenceEngine::details::convertFunctionToICNNNetwork(f, nGraphImpl, true),
                     InferenceEngine::details::InferenceEngineException)
                     << "failed node: " << ngraph_node->get_type_name() << std::endl;
        try {
            InferenceEngine::details::convertFunctionToICNNNetwork(f, nGraphImpl, true);
        } catch (InferenceEngine::details::InferenceEngineException &err) {
            std::string type_name = ngraph_node->get_type_name();

            std::map<std::string, std::string> exceptions = { {"Broadcast", "Tile"}, {"Interpolate", "Interp"},
                                                              {"NormalizeL2", "NormalizeIE"},
                                                              {"GroupConvolution", "ConvolutionIE"},
                                                              {"ConvolutionBackpropData", "DeconvolutionIE"},
                                                              {"GroupConvolutionBackpropData", "DeconvolutionIE"},
                                                              };
            std::string type_name_ie = type_name + "IE";
            if (exceptions[type_name].empty()) {
                type_name_ie = type_name + "IE";
            } else {
                type_name_ie = exceptions[type_name];
            }
            std::string expected_error_message = type_name + " operation has a form that is not supported. "
                    + ngraph_node->get_friendly_name()  + " should be converted to " + type_name_ie + " operation.";
            std::string real_message = err.what();
            bool is_messages_match = real_message.find(expected_error_message) != std::string::npos;
            EXPECT_TRUE(is_messages_match) << "failed node: " << type_name << std::endl
                        << "Exception massage: " << err.what() << std::endl
                        << "Expected message: " << expected_error_message << std:: endl;
        } catch (...) {
            FAIL() << "ERROR: Unexpected exception thrown: " << std::current_exception << std::endl;
        }
    }
}

TEST(ConvertFunctionToCNNNetworkTests, ConvertTopKWithOneInput) {
    std::shared_ptr<ngraph::Function> f;
    {
        auto param = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 22, 22});
        ngraph::Shape const_shape = {};
        std::vector<int64_t> val = {5};
        auto k = std::make_shared<ngraph::opset4::Constant>(ngraph::element::i64, const_shape, val);
        auto topK = std::make_shared<ngraph::opset4::TopK>(param, k, 2, ngraph::opset4::TopK::Mode::MAX, ngraph::opset4::TopK::SortType::SORT_VALUES);
        topK->set_friendly_name("topK");
        auto result = std::make_shared<ngraph::op::Result>(topK->output(1));

        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                               ngraph::ParameterVector{param});
        ngraph::pass::InitNodeInfo().run_on_function(f);
    }

    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::CommonOptimizations>();
    manager.register_pass<ngraph::pass::ConvertOpSet3ToOpSet2>();
    manager.register_pass<ngraph::pass::ConvertOpSet2ToOpSet1>();

    std::vector<std::pair<ngraph::element::Type, ngraph::element::Type>> convert_precision_list {
            {ngraph::element::i64, ngraph::element::i32},
            {ngraph::element::u64, ngraph::element::i32},
            {ngraph::element::u16, ngraph::element::i32},
            {ngraph::element::u32, ngraph::element::i32},
            {ngraph::element::f16, ngraph::element::f32},
            {ngraph::element::boolean, ngraph::element::u8},
    };

    for (auto & precision : convert_precision_list) {
        manager.register_pass<ngraph::pass::ConvertPrecision>(precision.first, precision.second);
    }

    manager.register_pass<ngraph::pass::ConvertOpSet1ToLegacy>();
    manager.register_pass<ngraph::pass::ConvertPrecision>(ngraph::element::i64, ngraph::element::i32);

    manager.run_passes(f);

    InferenceEngine::CNNNetwork nGraphImpl(f);
    nGraphImpl = CNNNetwork(InferenceEngine::details::convertFunctionToICNNNetwork(f, nGraphImpl));

    try {
        OutputsDataMap outputs = nGraphImpl.getOutputsInfo();
        ASSERT_EQ(outputs.size(), 1);
        ASSERT_EQ(outputs.begin()->first, "topK.1");
    } catch (InferenceEngine::details::InferenceEngineException &err) {
        const std::string ref_msg = "Error of validate layer: prelu with type: PReLU. Number of inputs (2) is not equal to expected ones: 1";
        const std::string resp_msg = err.what();
        ASSERT_TRUE(resp_msg.find(ref_msg) != std::string::npos) << resp_msg;
    }
}