// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cpp/ie_cnn_network.h>
#include <gtest/gtest.h>
#include <legacy/net_pass.h>

#include <common_test_utils/ov_test_utils.hpp>
#include <fstream>
#include <ie_core.hpp>
#include <ie_parameter.hpp>
#include <legacy/cnn_network_impl.hpp>
#include <legacy/convert_function_to_cnn_network.hpp>
#include <legacy/details/ie_cnn_network_iterator.hpp>
#include <legacy/ie_util_internal.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp>
#include <map>
#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph/op/convert.hpp>
#include <ngraph/op/maximum.hpp>
#include <ngraph/op/parameter.hpp>
#include <ngraph/op/prelu.hpp>
#include <ngraph/op/relu.hpp>
#include <ngraph/op/result.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include <openvino/core/model.hpp>
#include <openvino/core/node_vector.hpp>
#include <sstream>
#include <string>

#include "cnn_network_ngraph_impl.hpp"
#include "cnnlayer_utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "ie_precision.hpp"
#include "transformations/rt_info/primitives_priority_attribute.hpp"

using namespace testing;
using namespace InferenceEngine;

TEST(CNNNGraphImplTests, TestReshapeWithSameShape) {
    std::shared_ptr<ngraph::Function> f;
    {
        auto input = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1000, 4});
        input->set_friendly_name("input");
        auto shape = ngraph::opset5::Constant::create(ngraph::element::i64, {2}, {1, 4000});
        auto reshape = std::make_shared<ngraph::opset5::Reshape>(input, shape, true);
        f = std::make_shared<ngraph::Function>(ngraph::OutputVector{reshape}, ngraph::ParameterVector{input});
    }

    auto net = InferenceEngine::CNNNetwork(f);
    ASSERT_NO_THROW(net.reshape({{"input", SizeVector({1, 4000})}}));
}

TEST(CNNNGraphImplTests, TestTwoResultsFromOneTensor) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        auto result1 = std::make_shared<ngraph::op::Result>(relu);
        auto result2 = std::make_shared<ngraph::op::Result>(relu);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result1, result2};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    InferenceEngine::CNNNetwork cnnNet(ngraph);
    ASSERT_NO_THROW(auto convertedNet = std::make_shared<details::CNNNetworkImpl>(cnnNet));
}

TEST(CNNNGraphImplTests, TestInvalidReshape) {
    std::shared_ptr<ngraph::Function> f;
    {
        auto input = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1000, 4});
        input->set_friendly_name("input");
        auto shape = ngraph::opset5::Constant::create(ngraph::element::i64, {2}, {1, 4000});
        auto reshape = std::make_shared<ngraph::opset5::Reshape>(input, shape, true);
        f = std::make_shared<ngraph::Function>(ngraph::OutputVector{reshape}, ngraph::ParameterVector{input});
    }

    auto net = InferenceEngine::CNNNetwork(f);
    ASSERT_ANY_THROW(net.reshape({{"input", SizeVector({4})}}));

    auto param = *net.getFunction()->get_parameters().begin();
    ASSERT_EQ(param->get_output_shape(0), ngraph::Shape({1, 1000, 4}));

    ASSERT_NO_THROW(net.reshape({{"input", SizeVector({1, 1000, 4})}}));
}

TEST(CNNNGraphImplTests, TestNMS5OutputNames) {
    std::shared_ptr<ngraph::Function> f;
    {
        auto boxes = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1000, 4});
        auto scores = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1, 1000});
        auto max_output_boxes_per_class = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{}, {10});
        auto iou_threshold = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{}, {0.75});
        auto score_threshold = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{}, {0.7});
        auto nms = std::make_shared<ngraph::opset5::NonMaxSuppression>(
            boxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
            ngraph::opset5::NonMaxSuppression::BoxEncodingType::CORNER,
            true);
        nms->set_friendly_name("nms");
        f = std::make_shared<ngraph::Function>(ngraph::OutputVector{nms->output(0), nms->output(1), nms->output(2)},
                                               ngraph::ParameterVector{boxes, scores});
    }

    InferenceEngine::CNNNetwork cnnNet(f);
    auto outputs_info = cnnNet.getOutputsInfo();
    ASSERT_EQ(outputs_info.size(), 3);
    ASSERT_EQ(outputs_info.count("nms.0"), 1);
    ASSERT_EQ(outputs_info.count("nms.1"), 1);
    ASSERT_EQ(outputs_info.count("nms.2"), 1);
}

IE_SUPPRESS_DEPRECATED_START

TEST(CNNNGraphImplTests, TestConvertWithRemoveLastLayerNetwork) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        ngraph::element::Type type(ngraph::element::Type_t::i32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        param->set_friendly_name("param");
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        relu->set_friendly_name("relu");
        auto convert = std::make_shared<ngraph::op::Convert>(relu, ngraph::element::Type_t::i64);
        convert->set_friendly_name("convert");
        auto result = std::make_shared<ngraph::op::Result>(convert);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    InferenceEngine::CNNNetwork cnnNet(ngraph);
    auto convertedNet = InferenceEngine::CNNNetwork(std::make_shared<details::CNNNetworkImpl>(cnnNet));
    // Remove convert layer
    InferenceEngine::NetPass::ConvertPrecision(convertedNet, Precision::I64, Precision::I32);
    ASSERT_NO_THROW(cloneNet(convertedNet));
}

TEST(CNNNGraphImplTests, TestResultWithNotEqualName) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        relu->set_friendly_name("test_layer_name");
        auto result = std::make_shared<ngraph::op::Result>(relu);
        result->set_friendly_name("test_layer_name");

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    InferenceEngine::CNNNetwork cnnNet(ngraph);
    ASSERT_NO_THROW(auto convertedNet = std::make_shared<details::CNNNetworkImpl>(cnnNet));
}

TEST(CNNNGraphImplTests, TestGetOutputAfterConvertNetwork) {
    const std::string testLayerName = "testReLU";
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        relu->set_friendly_name(testLayerName);
        auto relu2 = std::make_shared<ngraph::op::Relu>(relu);
        relu2->set_friendly_name("relu2");
        auto result = std::make_shared<ngraph::op::Result>(relu2);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    InferenceEngine::CNNNetwork cnnNet(ngraph);
    // convert to old representation
    InferenceEngine::CNNNetwork convertedNetwork(std::make_shared<details::CNNNetworkImpl>(cnnNet));
    convertedNetwork.addOutput(testLayerName);

    InferenceEngine::OutputsDataMap outs = cnnNet.getOutputsInfo();
    InferenceEngine::OutputsDataMap convertedOuts = convertedNetwork.getOutputsInfo();
    ASSERT_EQ(1, outs.size());
    ASSERT_EQ(2, convertedOuts.size());
}

TEST(CNNNGraphImplTests, TestSetCurrentBatch) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        auto result = std::make_shared<ngraph::op::Result>(relu);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    InferenceEngine::details::CNNNetworkNGraphImpl cnnNet(ngraph);
    ASSERT_EQ(1, cnnNet.getBatchSize());
    ASSERT_EQ(OK, cnnNet.setBatchSize(1, nullptr));
    ASSERT_EQ(1, cnnNet.getBatchSize());
    ASSERT_NE(nullptr, cnnNet.getFunction());
}

TEST(CNNNGraphImplTests, TestSetBatch) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        auto result = std::make_shared<ngraph::op::Result>(relu);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    InferenceEngine::details::CNNNetworkNGraphImpl cnnNet(ngraph);
    ASSERT_EQ(1, cnnNet.getBatchSize());
    ASSERT_EQ(OK, cnnNet.setBatchSize(2, nullptr));  // must not trigger conversion
    ASSERT_EQ(2, cnnNet.getBatchSize());
    ASSERT_NE(nullptr, cnnNet.getFunction());
}

TEST(CNNNGraphImplTests, TestGetBatchScalar) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::Shape shape({});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        auto result = std::make_shared<ngraph::op::Result>(relu);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    InferenceEngine::details::CNNNetworkNGraphImpl cnnNet(ngraph);
    ASSERT_EQ(1, cnnNet.getBatchSize());
}

TEST(CNNNGraphImplTests, TestSetBatchScalar) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::Shape shape({});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        auto result = std::make_shared<ngraph::op::Result>(relu);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    InferenceEngine::details::CNNNetworkNGraphImpl cnnNet(ngraph);
    ASSERT_EQ(1, cnnNet.getBatchSize());
    ASSERT_EQ(PARAMETER_MISMATCH, cnnNet.setBatchSize(2, nullptr));  // must not trigger conversion
}

TEST(CNNNGraphImplTests, TestGetBatchDynamic) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        auto param = std::make_shared<ngraph::op::Parameter>(ngraph::element::Type_t::f32,
                                                             ngraph::PartialShape{5, ngraph::Dimension::dynamic()});
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        auto result = std::make_shared<ngraph::op::Result>(relu);
        ngraph = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param});
    }

    InferenceEngine::details::CNNNetworkNGraphImpl cnnNet(ngraph);
    ASSERT_TRUE(cnnNet.getFunction()->get_parameters()[0]->get_partial_shape().is_dynamic());
    ASSERT_EQ(5, cnnNet.getBatchSize());
}

TEST(CNNNGraphImplTests, TestSetBatchDynamic) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        auto param =
            std::make_shared<ngraph::op::Parameter>(ngraph::element::Type_t::f32, ngraph::PartialShape::dynamic());
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        auto result = std::make_shared<ngraph::op::Result>(relu);
        ngraph = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param});
    }

    InferenceEngine::details::CNNNetworkNGraphImpl cnnNet(ngraph);
    ASSERT_EQ(1, cnnNet.getBatchSize());
    ASSERT_EQ(PARAMETER_MISMATCH, cnnNet.setBatchSize(2, nullptr));  // must not trigger conversion
}

TEST(CNNNGraphImplTests, TestDoesChangePrecisionsWithNewAPI) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        auto param =
            std::make_shared<ngraph::op::Parameter>(ngraph::element::Type_t::f16, ngraph::PartialShape::dynamic());
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        auto result = std::make_shared<ngraph::op::Result>(relu);
        ngraph = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param});
    }

    // new OpenVINO 2.0
    {
        auto ngraphImpl = std::make_shared<InferenceEngine::details::CNNNetworkNGraphImpl>(
            ngraph,
            std::vector<InferenceEngine::IExtensionPtr>{},
            true);
        InferenceEngine::CNNNetwork cnnNet(ngraphImpl);
        ASSERT_EQ(InferenceEngine::Precision::FP16,
                  cnnNet.getInputsInfo().begin()->second->getTensorDesc().getPrecision());
        ASSERT_EQ(InferenceEngine::Precision::FP16,
                  cnnNet.getOutputsInfo().begin()->second->getTensorDesc().getPrecision());
    }

    // current API
    {
        auto ngraphImpl = std::make_shared<InferenceEngine::details::CNNNetworkNGraphImpl>(ngraph);
        InferenceEngine::CNNNetwork cnnNet(ngraphImpl);
        ASSERT_EQ(InferenceEngine::Precision::FP32,
                  cnnNet.getInputsInfo().begin()->second->getTensorDesc().getPrecision());
        ASSERT_EQ(InferenceEngine::Precision::FP32,
                  cnnNet.getOutputsInfo().begin()->second->getTensorDesc().getPrecision());
    }
}

TEST(CNNNGraphImplTests, TestSaveAffinity) {
    const std::string testAffinity = "testAffinity";
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        auto& rtInfo = relu->get_rt_info();
        rtInfo["affinity"] = testAffinity;
        relu->set_friendly_name("testReLU");
        auto result = std::make_shared<ngraph::op::Result>(relu);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    InferenceEngine::CNNNetwork cnnNet(ngraph);
    auto convertedNetwork = std::make_shared<InferenceEngine::details::CNNNetworkImpl>(cnnNet);
    auto cnnLayer = CommonTestUtils::getLayerByName(InferenceEngine::CNNNetwork(convertedNetwork), "testReLU");
    ASSERT_NE(nullptr, cnnLayer);
    ASSERT_EQ(cnnLayer->affinity, testAffinity);
}

TEST(CNNNGraphImplTests, TestAddOutput) {
    const std::string testLayerName = "testReLU";
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        relu->set_friendly_name(testLayerName);
        auto relu2 = std::make_shared<ngraph::op::Relu>(relu);
        relu2->set_friendly_name("relu2");
        auto result = std::make_shared<ngraph::op::Result>(relu2);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    InferenceEngine::CNNNetwork cnnNet(ngraph);
    ASSERT_NE(nullptr, cnnNet.getFunction());
    ASSERT_EQ(4, cnnNet.layerCount());

    cnnNet.addOutput(testLayerName);
    ASSERT_NE(nullptr, cnnNet.getFunction());
    ASSERT_EQ(5, cnnNet.layerCount());
    auto outputs = cnnNet.getOutputsInfo();
    ASSERT_EQ(2, outputs.size());
    ASSERT_TRUE(outputs.find("relu2") != outputs.end());
    ASSERT_TRUE(outputs.find(testLayerName) != outputs.end());
}

TEST(CNNNGraphImplTests, TestAddOutputWithPort) {
    const std::string testLayerName = "testReLU";
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        relu->set_friendly_name(testLayerName);
        auto relu2 = std::make_shared<ngraph::op::Relu>(relu);
        relu2->set_friendly_name("relu2");
        auto result = std::make_shared<ngraph::op::Result>(relu2);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    InferenceEngine::CNNNetwork cnnNet(ngraph);
    ASSERT_NE(nullptr, cnnNet.getFunction());
    ASSERT_EQ(4, cnnNet.layerCount());

    cnnNet.addOutput(testLayerName, 0);
    ASSERT_NE(nullptr, cnnNet.getFunction());
    ASSERT_EQ(5, cnnNet.layerCount());

    EXPECT_THROW(cnnNet.addOutput(testLayerName, 1), OutOfBounds);
    ASSERT_NE(nullptr, cnnNet.getFunction());
    ASSERT_EQ(5, cnnNet.layerCount());

    auto outputs = cnnNet.getOutputsInfo();
    ASSERT_EQ(2, outputs.size());
    ASSERT_TRUE(outputs.find("relu2") != outputs.end());
    ASSERT_TRUE(outputs.find(testLayerName) != outputs.end());
}

TEST(CNNNGraphImplTests, TestAddOutputTwoTimes) {
    const std::string testLayerName = "testReLU";
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        relu->set_friendly_name(testLayerName);
        auto relu2 = std::make_shared<ngraph::op::Relu>(relu);
        relu2->set_friendly_name("relu2");
        auto result = std::make_shared<ngraph::op::Result>(relu2);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    InferenceEngine::CNNNetwork cnnNet(ngraph);
    ASSERT_NE(nullptr, cnnNet.getFunction());
    ASSERT_EQ(4, cnnNet.layerCount());

    cnnNet.addOutput(testLayerName);
    ASSERT_NE(nullptr, cnnNet.getFunction());
    ASSERT_EQ(5, cnnNet.layerCount());
    cnnNet.addOutput(testLayerName);
    ASSERT_NE(nullptr, cnnNet.getFunction());
    ASSERT_EQ(5, cnnNet.layerCount());
    auto outputs = cnnNet.getOutputsInfo();
    ASSERT_EQ(2, outputs.size());
    ASSERT_TRUE(outputs.find("relu2") != outputs.end());
    ASSERT_TRUE(outputs.find(testLayerName) != outputs.end());
}

TEST(CNNNGraphImplTests, TestAddOutputFromConvertedNetwork) {
    const std::string testLayerName = "testReLU";
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        relu->set_friendly_name(testLayerName);
        auto relu2 = std::make_shared<ngraph::op::Relu>(relu);
        relu2->set_friendly_name("relu2");
        auto result = std::make_shared<ngraph::op::Result>(relu2);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    InferenceEngine::CNNNetwork cnnNet(ngraph);
    ASSERT_NE(nullptr, cnnNet.getFunction());
    ASSERT_EQ(4, cnnNet.layerCount());

    cnnNet.addOutput(testLayerName);
    ASSERT_NE(nullptr, cnnNet.getFunction());
    ASSERT_EQ(5, cnnNet.layerCount());
    // convert to old representation
    InferenceEngine::CNNNetwork convertedNetwork(std::make_shared<InferenceEngine::details::CNNNetworkImpl>(cnnNet));
    auto outputs = convertedNetwork.getOutputsInfo();
    ASSERT_EQ(2, outputs.size());
    ASSERT_TRUE(outputs.find("relu2") != outputs.end());
    ASSERT_TRUE(outputs.find(testLayerName) != outputs.end());
}

TEST(CNNNGraphImplTests, ConstantAsInternalAndExternalLayer) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        auto constant = ngraph::op::Constant::create(ngraph::element::Type_t::f32, {1}, {2});
        auto prelu = std::make_shared<ngraph::op::PRelu>(param, constant);
        auto add = std::make_shared<ngraph::op::v1::Maximum>(prelu, constant);
        auto result = std::make_shared<ngraph::op::Result>(add);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    InferenceEngine::CNNNetwork cnnNet(ngraph);
    auto convertedNetwork = std::make_shared<InferenceEngine::details::CNNNetworkImpl>(cnnNet);
    ASSERT_EQ(4, convertedNetwork->layerCount());
}

TEST(CNNNGraphImplTests, SaveInputInfoAfterConversion) {
    std::string name = "param";
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        param->set_friendly_name(name);
        auto constant = ngraph::op::Constant::create(ngraph::element::Type_t::f32, {1}, {2});
        auto prelu = std::make_shared<ngraph::op::PRelu>(param, constant);
        auto add = std::make_shared<ngraph::op::v1::Maximum>(prelu, constant);
        auto result = std::make_shared<ngraph::op::Result>(add);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    InferenceEngine::CNNNetwork cnnNet(ngraph);
    auto inputInfo = cnnNet.getInputsInfo()[name];
    ASSERT_EQ(inputInfo->getPreProcess().getResizeAlgorithm(), ResizeAlgorithm::NO_RESIZE);
    inputInfo->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_AREA);
    ASSERT_EQ(inputInfo->getPreProcess().getResizeAlgorithm(), ResizeAlgorithm::RESIZE_AREA);

    auto cnnNetImpl = std::make_shared<details::CNNNetworkImpl>(cnnNet);
    inputInfo = cnnNetImpl->getInput(name);
    ASSERT_EQ(inputInfo->getPreProcess().getResizeAlgorithm(), ResizeAlgorithm::RESIZE_AREA);
}

TEST(CNNNGraphImplTests, SavePrimitivesPriority) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data shape="1,3,22,22" element_type="f32" PrimitivesPriority="cpu:avx2"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="ReLU" version="opset1">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    const Core ie;
    Blob::Ptr weights;

    auto network = ie.ReadNetwork(model, weights);
    auto inputInfo = network.getInputsInfo();
    auto convertedNetwork = std::make_shared<InferenceEngine::details::CNNNetworkImpl>(network);
    auto cnnLayer = getCreatorLayer(inputInfo.begin()->second->getInputData()).lock();
    ASSERT_NE(nullptr, cnnLayer);
    ASSERT_NE(cnnLayer->params.find("PrimitivesPriority"), cnnLayer->params.end());
    ASSERT_EQ("cpu:avx2", cnnLayer->params["PrimitivesPriority"]);
}

TEST(CNNNGraphImplTests, ReadFromCNNNetReader) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data shape="1,3,22,22" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="ReLU" version="opset1">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    InferenceEngine::Core core;
    CNNNetwork network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr());
    ASSERT_EQ(3, network.layerCount());
}

TEST(CNNNGraphImplTests, ReadMeanImageFromCNNNetReader) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <pre-process mean-precision="FP32" reference-layer-name="data">
        <channel id="0">
            <mean offset="0" size="1936"/>
        </channel>
        <channel id="1">
            <mean offset="1936" size="1936"/>
        </channel>
        <channel id="2">
            <mean offset="3872" size="1936"/>
        </channel>
    </pre-process>
    <layers>
        <layer name="data" type="Parameter" id="0" version="opset1">
            <data shape="1,3,22,22" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="ReLU" version="opset1">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    InferenceEngine::Core core;
    size_t hwSize = 22 * 22;
    size_t dataSize = hwSize * 3;
    Blob::Ptr weights = make_shared_blob<float>(TensorDesc(Precision::FP32, {dataSize}, Layout::C));
    weights->allocate();
    {
        auto lockData = weights->buffer();
        float* dataPtr = lockData.as<float*>();

        for (size_t i = 0; i < dataSize; ++i) {
            dataPtr[i] = 1;
        }
    }
    CNNNetwork network = core.ReadNetwork(model, weights);
    auto f = network.getFunction();

    std::shared_ptr<ngraph::Function> f_ref;
    auto data = std::make_shared<ov::op::v0::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 22, 22});
    {
        auto mean_image = ov::op::v0::Constant::create(ngraph::element::f32, ngraph::Shape{3, 22, 22}, {1});
        auto sub = std::make_shared<ov::op::v1::Subtract>(data, mean_image);
        auto relu = std::make_shared<ov::op::v0::Relu>(sub);
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{relu}, ngraph::ParameterVector{data});
    }

    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::ATTRIBUTES)
                        .enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(f, f_ref);
    EXPECT_TRUE(res.valid) << res.message;
}

TEST(CNNNGraphImplTests, ReadMeanValueFromCNNNetReader) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <pre-process mean-precision="FP32" reference-layer-name="data">
        <channel id="0">
            <mean value="1.1"/>
        </channel>
        <channel id="1">
            <mean value="2.2"/>
        </channel>
        <channel id="2">
            <mean value="3.3"/>
        </channel>
    </pre-process>
    <layers>
        <layer name="data" type="Parameter" id="0" version="opset1">
            <data shape="1,3,22,22" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="ReLU" version="opset1">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    InferenceEngine::Core core;
    Blob::Ptr weights{nullptr};
    CNNNetwork network = core.ReadNetwork(model, weights);
    auto f = network.getFunction();

    std::shared_ptr<ngraph::Function> f_ref;
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 22, 22});
        auto mean_image = ov::op::v0::Constant::create(ngraph::element::f32, ngraph::Shape{3, 1, 1}, {1.1, 2.2, 3.3});
        auto sub = std::make_shared<ov::op::v1::Subtract>(data, mean_image);
        auto relu = std::make_shared<ov::op::v0::Relu>(sub);
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{relu}, ngraph::ParameterVector{data});
    }

    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::ATTRIBUTES)
                        .enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(f, f_ref);
    EXPECT_TRUE(res.valid) << res.message;
}

TEST(CNNNGraphImplTests, CanChangeInputPrecision) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 16, 16});
        ngraph::element::Type type(ngraph::element::Type_t::f16);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        param->set_friendly_name("input");
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        relu->set_friendly_name("output");
        auto result = std::make_shared<ngraph::op::Result>(relu);
        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};
        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    InferenceEngine::CNNNetwork cnnNet(ngraph);
    {
        SCOPED_TRACE("After ctor");

        const auto inputsInfo = cnnNet.getInputsInfo();

        ASSERT_EQ(inputsInfo.at("input")->getPrecision(), Precision::FP32) << "FP32 is default presision";
    }
    {
        SCOPED_TRACE("Manually set input precision");

        const auto inputsInfo = cnnNet.getInputsInfo();

        inputsInfo.at("input")->setPrecision(Precision::FP16);
    }
    InferenceEngine::CNNNetwork convertedNetwork;
    {
        SCOPED_TRACE("Convert to old format");

        // convert to old representation
        convertedNetwork =
            InferenceEngine::CNNNetwork(std::make_shared<InferenceEngine::details::CNNNetworkImpl>(cnnNet));
    }
    {
        SCOPED_TRACE("After conversion");

        const auto inputsInfo = convertedNetwork.getInputsInfo();

        ASSERT_EQ(inputsInfo.at("input")->getPrecision(), Precision::FP16)
            << "Manually set presision should be left unchanged";
    }
}

TEST(CNNNGraphImplTests, CanChangeInputLayout) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 16, 16});
        ngraph::element::Type type(ngraph::element::Type_t::f16);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        param->set_friendly_name("input");
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        relu->set_friendly_name("output");
        auto result = std::make_shared<ngraph::op::Result>(relu);
        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};
        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    InferenceEngine::CNNNetwork cnnNet(ngraph);
    {
        SCOPED_TRACE("After ctor");

        const auto inputsInfo = cnnNet.getInputsInfo();

        ASSERT_EQ(inputsInfo.at("input")->getLayout(), Layout::NCHW) << "NCHW is default layout";
    }
    {
        SCOPED_TRACE("Manually set input layout");

        const auto inputsInfo = cnnNet.getInputsInfo();

        inputsInfo.at("input")->setLayout(Layout::NHWC);
    }
    InferenceEngine::CNNNetwork convertedNetwork;
    {
        SCOPED_TRACE("Convert to old format");

        // convert to old representation
        convertedNetwork =
            InferenceEngine::CNNNetwork(std::make_shared<InferenceEngine::details::CNNNetworkImpl>(cnnNet));
    }
    {
        SCOPED_TRACE("After conversion");

        const auto inputsInfo = convertedNetwork.getInputsInfo();

        ASSERT_EQ(inputsInfo.at("input")->getLayout(), Layout::NHWC) << "Manually set layout should be left unchanged";
    }
}

TEST(CNNNGraphImplTests, CanChangeOutputPrecision) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 16, 16});
        ngraph::element::Type type(ngraph::element::Type_t::f16);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        param->set_friendly_name("input");
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        relu->set_friendly_name("output");
        auto result = std::make_shared<ngraph::op::Result>(relu);
        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};
        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    InferenceEngine::CNNNetwork cnnNet(ngraph);
    {
        SCOPED_TRACE("After ctor");

        const auto outputsInfo = cnnNet.getOutputsInfo();

        ASSERT_EQ(outputsInfo.at("output")->getPrecision(), Precision::FP32) << "FP32 is default presision";
    }
    {
        SCOPED_TRACE("Manually set output precision");

        const auto outputsInfo = cnnNet.getOutputsInfo();

        outputsInfo.at("output")->setPrecision(Precision::FP16);
    }
    InferenceEngine::CNNNetwork convertedNetwork;
    {
        SCOPED_TRACE("Convert to old format");

        // convert to old representation
        convertedNetwork =
            InferenceEngine::CNNNetwork(std::make_shared<InferenceEngine::details::CNNNetworkImpl>(cnnNet));
    }
    {
        SCOPED_TRACE("After conversion");

        const auto outputsInfo = convertedNetwork.getOutputsInfo();

        ASSERT_EQ(outputsInfo.at("output")->getPrecision(), Precision::FP16)
            << "Manually set presision should be left unchanged";
    }
}

TEST(CNNNGraphImplTests, CanChangeOutputLayout) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 16, 16});
        ngraph::element::Type type(ngraph::element::Type_t::f16);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        param->set_friendly_name("input");
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        relu->set_friendly_name("output");
        auto result = std::make_shared<ngraph::op::Result>(relu);
        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};
        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    InferenceEngine::CNNNetwork cnnNet(ngraph);
    {
        SCOPED_TRACE("After ctor");

        const auto outputsInfo = cnnNet.getOutputsInfo();

        ASSERT_EQ(outputsInfo.at("output")->getLayout(), Layout::NCHW) << "NCHW is default layout";
    }
    {
        SCOPED_TRACE("Manually set output layout");

        const auto outputsInfo = cnnNet.getOutputsInfo();

        outputsInfo.at("output")->setLayout(Layout::NHWC);
    }
    InferenceEngine::CNNNetwork convertedNetwork;
    {
        SCOPED_TRACE("Convert to old format");

        // convert to old representation
        convertedNetwork =
            InferenceEngine::CNNNetwork(std::make_shared<InferenceEngine::details::CNNNetworkImpl>(cnnNet));
    }
    {
        SCOPED_TRACE("After conversion");

        const auto outputsInfo = convertedNetwork.getOutputsInfo();

        ASSERT_EQ(outputsInfo.at("output")->getLayout(), Layout::NHWC)
            << "Manually set layout should be left unchanged";
    }
}

TEST(CNNNGraphImplTests, TestCheckStats) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        auto result = std::make_shared<ngraph::op::Result>(relu);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    InferenceEngine::details::CNNNetworkNGraphImpl cnnNet(ngraph);
}

TEST(CNNNGraphImplTests, CanSetBatchReadValue) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        auto input = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2});
        auto constant = std::make_shared<ngraph::opset3::Constant>(ngraph::element::f32,
                                                                   ngraph::Shape{1, 2},
                                                                   std::vector<float>{1, 2});

        auto read_value = std::make_shared<ngraph::opset3::ReadValue>(constant, "variable_id");
        auto assign = std::make_shared<ngraph::opset3::Assign>(read_value, "variable_id");
        assign->add_control_dependency(read_value);
        auto add = std::make_shared<ngraph::opset3::Add>(input, read_value);
        auto result = std::make_shared<ngraph::op::Result>(add);

        ngraph::ParameterVector params = {input};
        ngraph::ResultVector results = {result};
        ngraph::SinkVector sinks = {assign};

        ngraph = std::make_shared<ngraph::Function>(results, sinks, params);
    }

    InferenceEngine::CNNNetwork cnnNet(ngraph);
    auto convertedNet = std::make_shared<details::CNNNetworkImpl>(cnnNet);
    auto status = convertedNet->setBatchSize(4, nullptr);
    EXPECT_EQ(status, StatusCode::OK);
}

TEST(CNNNGraphImplTests, addSameOutput) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::opset3::Parameter>(type, shape);
        auto relu = std::make_shared<ngraph::opset3::Relu>(param);
        auto shapeof = std::make_shared<ngraph::opset3::ShapeOf>(param);
        auto reshape = std::make_shared<ngraph::opset3::Reshape>(relu, shapeof, true);
        reshape->set_friendly_name("reshape");
        auto result = std::make_shared<ngraph::op::Result>(reshape);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    CNNNetwork cnnNetwork(ngraph);
    cnnNetwork.addOutput("reshape");
    auto outputs = cnnNetwork.getOutputsInfo();

    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(outputs.count("reshape"), 1);
    ASSERT_EQ(outputs["reshape"]->getLayout(), InferenceEngine::Layout::NCHW);
}

TEST(CNNNGraphImplTests, addOutput) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::opset3::Parameter>(type, shape);
        auto relu = std::make_shared<ngraph::opset3::Relu>(param);
        auto shapeof = std::make_shared<ngraph::opset3::ShapeOf>(param);
        auto reshape = std::make_shared<ngraph::opset3::Reshape>(relu, shapeof, true);
        reshape->set_friendly_name("reshape");
        auto relu2 = std::make_shared<ngraph::opset3::Relu>(reshape);
        auto result = std::make_shared<ngraph::op::Result>(relu2);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    CNNNetwork cnnNetwork(ngraph);
    cnnNetwork.addOutput("reshape");
    auto outputs = cnnNetwork.getOutputsInfo();

    ASSERT_EQ(outputs.size(), 2);
    ASSERT_EQ(outputs.count("reshape"), 1);
    ASSERT_EQ(outputs["reshape"]->getLayout(), InferenceEngine::Layout::NCHW);
}

TEST(CNNNGraphImplTests, addOutputForParameter) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::opset3::Parameter>(type, shape);
        param->set_friendly_name("param");
        auto relu = std::make_shared<ngraph::opset3::Relu>(param);
        auto result = std::make_shared<ngraph::op::Result>(relu);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    CNNNetwork cnnNetwork(ngraph);
    cnnNetwork.addOutput("param");
    {
        auto output_info = cnnNetwork.getOutputsInfo();
        ASSERT_EQ(output_info.count("param"), 1);
        ASSERT_EQ(output_info["param"]->getTensorDesc().getDims(), SizeVector({1, 3, 22, 22}));
    }

    cnnNetwork.reshape({{"param", SizeVector({1, 3, 32, 64})}});
    cnnNetwork.addOutput("param");
    {
        auto output_info = cnnNetwork.getOutputsInfo();
        ASSERT_EQ(output_info.count("param"), 1);
        ASSERT_EQ(output_info["param"]->getTensorDesc().getDims(), SizeVector({1, 3, 32, 64}));
    }
}

TEST(CNNNGraphImplTests, AddOutputToExperimentalOpOpset6) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer id="0" name="in0" type="Parameter" version="opset1">
            <data element_type="f32" shape="1000,4"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1000</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="in1" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,256,200,336"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>200</dim>
                    <dim>336</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="in2" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,256,100,168"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>100</dim>
                    <dim>168</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="in3" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,256,50,84"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>50</dim>
                    <dim>84</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="in4" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,256,25,42"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>25</dim>
                    <dim>42</dim>
                </port>
            </output>
        </layer>
        <layer id="5" name="exp" type="ExperimentalDetectronROIFeatureExtractor" version="opset6">
            <data aligned="0" output_size="7" pyramid_scales="4,8,16,32" sampling_ratio="2"/>
            <input>
                <port id="0">
                    <dim>1000</dim>
                    <dim>4</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>200</dim>
                    <dim>336</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>100</dim>
                    <dim>168</dim>
                </port>
                <port id="3">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>50</dim>
                    <dim>84</dim>
                </port>
                <port id="4">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>25</dim>
                    <dim>42</dim>
                </port>
            </input>
            <output>
                <port id="5" precision="FP32">
                    <dim>1000</dim>
                    <dim>256</dim>
                    <dim>7</dim>
                    <dim>7</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="6" type="ReLU" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1000</dim>
                    <dim>256</dim>
                    <dim>7</dim>
                    <dim>7</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32">
                    <dim>1000</dim>
                    <dim>256</dim>
                    <dim>7</dim>
                    <dim>7</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="7" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1000</dim>
                    <dim>256</dim>
                    <dim>7</dim>
                    <dim>7</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="5" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="5" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="5" to-port="2"/>
        <edge from-layer="3" from-port="0" to-layer="5" to-port="3"/>
        <edge from-layer="4" from-port="0" to-layer="5" to-port="4"/>
        <edge from-layer="5" from-port="5" to-layer="6" to-port="0"/>
        <edge from-layer="6" from-port="1" to-layer="7" to-port="0"/>
    </edges>
</net>
)V0G0N";
    InferenceEngine::Core core;
    CNNNetwork network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr());
    network.addOutput("exp");
    auto outputs = network.getOutputsInfo();
    ASSERT_NE(outputs.find("exp.0"), outputs.end());
}

TEST(CNNNGraphImplTests, AddOutputToExperimentalOp) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer id="0" name="in0" type="Parameter" version="opset1">
            <data element_type="f32" shape="1000,4"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1000</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="in1" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,256,200,336"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>200</dim>
                    <dim>336</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="in2" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,256,100,168"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>100</dim>
                    <dim>168</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="in3" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,256,50,84"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>50</dim>
                    <dim>84</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="in4" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,256,25,42"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>25</dim>
                    <dim>42</dim>
                </port>
            </output>
        </layer>
        <layer id="5" name="exp" type="ExperimentalDetectronROIFeatureExtractor" version="experimental">
            <data aligned="0" output_size="7" pyramid_scales="4,8,16,32" sampling_ratio="2"/>
            <input>
                <port id="0">
                    <dim>1000</dim>
                    <dim>4</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>200</dim>
                    <dim>336</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>100</dim>
                    <dim>168</dim>
                </port>
                <port id="3">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>50</dim>
                    <dim>84</dim>
                </port>
                <port id="4">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>25</dim>
                    <dim>42</dim>
                </port>
            </input>
            <output>
                <port id="5" precision="FP32">
                    <dim>1000</dim>
                    <dim>256</dim>
                    <dim>7</dim>
                    <dim>7</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="6" type="ReLU" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1000</dim>
                    <dim>256</dim>
                    <dim>7</dim>
                    <dim>7</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32">
                    <dim>1000</dim>
                    <dim>256</dim>
                    <dim>7</dim>
                    <dim>7</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="7" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1000</dim>
                    <dim>256</dim>
                    <dim>7</dim>
                    <dim>7</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="5" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="5" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="5" to-port="2"/>
        <edge from-layer="3" from-port="0" to-layer="5" to-port="3"/>
        <edge from-layer="4" from-port="0" to-layer="5" to-port="4"/>
        <edge from-layer="5" from-port="5" to-layer="6" to-port="0"/>
        <edge from-layer="6" from-port="1" to-layer="7" to-port="0"/>
    </edges>
</net>
)V0G0N";
    InferenceEngine::Core core;
    CNNNetwork network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr());
    network.addOutput("exp");
    auto outputs = network.getOutputsInfo();
    ASSERT_NE(outputs.find("exp.0"), outputs.end());
}

TEST(CNNNGraphImplTests, SaveOriginalResultNameForMultiOutputOp) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer id="0" name="in0" type="Parameter" version="opset1">
            <data element_type="f32" shape="1000,4"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1000</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="in1" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,256,200,336"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>200</dim>
                    <dim>336</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="in2" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,256,100,168"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>100</dim>
                    <dim>168</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="in3" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,256,50,84"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>50</dim>
                    <dim>84</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="in4" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,256,25,42"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>25</dim>
                    <dim>42</dim>
                </port>
            </output>
        </layer>
        <layer id="5" name="exp" type="ExperimentalDetectronROIFeatureExtractor" version="experimental">
            <data aligned="0" output_size="7" pyramid_scales="4,8,16,32" sampling_ratio="2"/>
            <input>
                <port id="0">
                    <dim>1000</dim>
                    <dim>4</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>200</dim>
                    <dim>336</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>100</dim>
                    <dim>168</dim>
                </port>
                <port id="3">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>50</dim>
                    <dim>84</dim>
                </port>
                <port id="4">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>25</dim>
                    <dim>42</dim>
                </port>
            </input>
            <output>
                <port id="5" precision="FP32">
                    <dim>1000</dim>
                    <dim>256</dim>
                    <dim>7</dim>
                    <dim>7</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="fake_const" type="Const" version="opset1">
            <data offset="0" size="4" shape="1,1,1,1" element_type="f32"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="7" name="text_features" type="Add" version="opset1">
            <input>
                <port id="0">
                    <dim>1000</dim>
                    <dim>256</dim>
                    <dim>7</dim>
                    <dim>7</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1000</dim>
                    <dim>256</dim>
                    <dim>7</dim>
                    <dim>7</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="8" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1000</dim>
                    <dim>256</dim>
                    <dim>7</dim>
                    <dim>7</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="5" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="5" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="5" to-port="2"/>
        <edge from-layer="3" from-port="0" to-layer="5" to-port="3"/>
        <edge from-layer="4" from-port="0" to-layer="5" to-port="4"/>
        <edge from-layer="5" from-port="5" to-layer="7" to-port="0"/>
        <edge from-layer="6" from-port="1" to-layer="7" to-port="1"/>
        <edge from-layer="7" from-port="2" to-layer="8" to-port="0"/>
    </edges>
</net>
)V0G0N";

    InferenceEngine::Core core;
    Blob::Ptr data = make_shared_blob<float>(TensorDesc(Precision::FP32, {4}, Layout::C));
    data->allocate();
    {
        auto lockData = data->buffer();
        float* dataPtr = lockData.as<float*>();

        for (size_t i = 0; i < 4; ++i) {
            dataPtr[i] = 0;
        }
    }
    CNNNetwork network = core.ReadNetwork(model, data);
    {
        auto outputs = network.getOutputsInfo();
        ASSERT_NE(outputs.find("text_features"), outputs.end());
    }

    auto nGraphFunc = network.getFunction();

    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::ConvertOpSet1ToLegacy>();
    manager.run_passes(nGraphFunc);

    auto clonedNetwork = InferenceEngine::details::convertFunctionToICNNNetwork(nGraphFunc, network);
    {
        OutputsDataMap outputs;
        clonedNetwork->getOutputsInfo(outputs);
        ASSERT_NE(outputs.find("text_features"), outputs.end());
    }
}

TEST(CNNNGraphImplTests, SaveOriginalResultNameForMultiOutputOpOpset6) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer id="0" name="in0" type="Parameter" version="opset1">
            <data element_type="f32" shape="1000,4"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1000</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="in1" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,256,200,336"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>200</dim>
                    <dim>336</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="in2" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,256,100,168"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>100</dim>
                    <dim>168</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="in3" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,256,50,84"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>50</dim>
                    <dim>84</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="in4" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,256,25,42"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>25</dim>
                    <dim>42</dim>
                </port>
            </output>
        </layer>
        <layer id="5" name="exp" type="ExperimentalDetectronROIFeatureExtractor" version="opset6">
            <data aligned="0" output_size="7" pyramid_scales="4,8,16,32" sampling_ratio="2"/>
            <input>
                <port id="0">
                    <dim>1000</dim>
                    <dim>4</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>200</dim>
                    <dim>336</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>100</dim>
                    <dim>168</dim>
                </port>
                <port id="3">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>50</dim>
                    <dim>84</dim>
                </port>
                <port id="4">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>25</dim>
                    <dim>42</dim>
                </port>
            </input>
            <output>
                <port id="5" precision="FP32">
                    <dim>1000</dim>
                    <dim>256</dim>
                    <dim>7</dim>
                    <dim>7</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="fake_const" type="Const" version="opset1">
            <data offset="0" size="4" shape="1,1,1,1" element_type="f32"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="7" name="text_features" type="Add" version="opset1">
            <input>
                <port id="0">
                    <dim>1000</dim>
                    <dim>256</dim>
                    <dim>7</dim>
                    <dim>7</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1000</dim>
                    <dim>256</dim>
                    <dim>7</dim>
                    <dim>7</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="8" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1000</dim>
                    <dim>256</dim>
                    <dim>7</dim>
                    <dim>7</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="5" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="5" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="5" to-port="2"/>
        <edge from-layer="3" from-port="0" to-layer="5" to-port="3"/>
        <edge from-layer="4" from-port="0" to-layer="5" to-port="4"/>
        <edge from-layer="5" from-port="5" to-layer="7" to-port="0"/>
        <edge from-layer="6" from-port="1" to-layer="7" to-port="1"/>
        <edge from-layer="7" from-port="2" to-layer="8" to-port="0"/>
    </edges>
</net>
)V0G0N";

    InferenceEngine::Core core;
    Blob::Ptr data = make_shared_blob<float>(TensorDesc(Precision::FP32, {4}, Layout::C));
    data->allocate();
    {
        auto lockData = data->buffer();
        float* dataPtr = lockData.as<float*>();

        for (size_t i = 0; i < 4; ++i) {
            dataPtr[i] = 0;
        }
    }
    CNNNetwork network = core.ReadNetwork(model, data);
    {
        auto outputs = network.getOutputsInfo();
        ASSERT_NE(outputs.find("text_features"), outputs.end());
    }

    auto nGraphFunc = network.getFunction();

    ngraph::pass::Manager manager;

    manager.register_pass<ngraph::pass::ConvertOpSet1ToLegacy>();

    manager.run_passes(nGraphFunc);

    auto clonedNetwork = InferenceEngine::details::convertFunctionToICNNNetwork(nGraphFunc, network);
    {
        OutputsDataMap outputs;
        clonedNetwork->getOutputsInfo(outputs);
        ASSERT_NE(outputs.find("text_features"), outputs.end());
    }
}

TEST(CNNNGraphImplTests, CheckUniqueNames) {
    std::shared_ptr<ngraph::Function> f;
    {
        auto boxes = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1000, 4});
        boxes->set_friendly_name("boxes");
        auto scores = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1, 1000});
        scores->set_friendly_name("scores");
        auto max_output_boxes_per_class = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{}, {10});
        auto iou_threshold = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{}, {0.75});
        auto score_threshold = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{}, {0.7});
        auto nms = std::make_shared<ngraph::opset5::NonMaxSuppression>(
            boxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
            ngraph::opset5::NonMaxSuppression::BoxEncodingType::CORNER,
            true);

        auto result1 = std::make_shared<ngraph::opset5::Result>(nms->output(0));
        result1->set_friendly_name("result1");
        auto result2 = std::make_shared<ngraph::opset5::Result>(nms->output(1));
        result2->set_friendly_name("result2");
        auto result3 = std::make_shared<ngraph::opset5::Result>(nms->output(2));
        result3->set_friendly_name("result3");
        nms->set_friendly_name("nms");
        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{result1, result2, result3},
                                               ngraph::ParameterVector{boxes, scores});
    }

    ASSERT_NO_THROW(InferenceEngine::CNNNetwork{f});
}

TEST(CNNNGraphImplTests, CheckNonUniqueParameterName) {
    std::shared_ptr<ngraph::Function> f;
    {
        auto boxes = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1000, 4});
        boxes->set_friendly_name("boxes");
        auto scores = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1, 1000});
        scores->set_friendly_name("boxes");
        auto max_output_boxes_per_class = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{}, {10});
        auto iou_threshold = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{}, {0.75});
        auto score_threshold = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{}, {0.7});
        auto nms = std::make_shared<ngraph::opset5::NonMaxSuppression>(
            boxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
            ngraph::opset5::NonMaxSuppression::BoxEncodingType::CORNER,
            true);

        auto result1 = std::make_shared<ngraph::opset5::Result>(nms->output(0));
        result1->set_friendly_name("result1");
        auto result2 = std::make_shared<ngraph::opset5::Result>(nms->output(1));
        result2->set_friendly_name("result2");
        auto result3 = std::make_shared<ngraph::opset5::Result>(nms->output(2));
        result3->set_friendly_name("result3");
        nms->set_friendly_name("nms");
        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{result1, result2, result3},
                                               ngraph::ParameterVector{boxes, scores});
    }

    ASSERT_THROW(InferenceEngine::CNNNetwork{f}, InferenceEngine::Exception);
}

TEST(CNNNGraphImplTests, CheckNonUniqueResultName) {
    std::shared_ptr<ngraph::Function> f;
    {
        auto boxes = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1000, 4});
        boxes->set_friendly_name("nms.1");
        auto scores = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1, 1000});
        scores->set_friendly_name("scores");
        auto max_output_boxes_per_class = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{}, {10});
        auto iou_threshold = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{}, {0.75});
        auto score_threshold = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{}, {0.7});
        auto nms = std::make_shared<ngraph::opset5::NonMaxSuppression>(
            boxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
            ngraph::opset5::NonMaxSuppression::BoxEncodingType::CORNER,
            true);

        auto result1 = std::make_shared<ngraph::opset5::Result>(nms->output(0));
        result1->set_friendly_name("result1");
        auto result2 = std::make_shared<ngraph::opset5::Result>(nms->output(1));
        result2->set_friendly_name("result2");
        auto result3 = std::make_shared<ngraph::opset5::Result>(nms->output(2));
        result3->set_friendly_name("result3");
        nms->set_friendly_name("nms");
        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{result1, result2, result3},
                                               ngraph::ParameterVector{boxes, scores});
    }

    ASSERT_THROW(InferenceEngine::CNNNetwork{f}, InferenceEngine::Exception);
}

TEST(CNNNGraphImplTests, CheckNonUniqueNewResultName) {
    std::shared_ptr<ngraph::Function> f;
    {
        auto boxes = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1000, 4});
        boxes->set_friendly_name("nms.1");
        auto scores = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1, 1000});
        scores->set_friendly_name("scores");
        auto max_output_boxes_per_class = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{}, {10});
        auto iou_threshold = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{}, {0.75});
        auto score_threshold = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{}, {0.7});
        auto nms = std::make_shared<ngraph::opset5::NonMaxSuppression>(
            boxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
            ngraph::opset5::NonMaxSuppression::BoxEncodingType::CORNER,
            true);

        auto result1 = std::make_shared<ngraph::opset5::Result>(nms->output(0));
        result1->set_friendly_name("result1");
        auto result3 = std::make_shared<ngraph::opset5::Result>(nms->output(2));
        result3->set_friendly_name("result3");
        nms->set_friendly_name("nms");
        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{result1, result3},
                                               ngraph::ParameterVector{boxes, scores});
    }

    CNNNetwork cnnNet;
    ASSERT_NO_THROW(cnnNet = InferenceEngine::CNNNetwork{f});
    ASSERT_THROW(cnnNet.addOutput("nms", 1), InferenceEngine::Exception);
}

TEST(CNNNGraphImplTests, RemoveLoopDanglingParametersIfConcatEmptyTensor) {
    CNNNetwork network;

    auto trip_count = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, 10);
    auto condition = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{}, true);

    auto a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});
    auto ai = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});
    auto b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2});
    auto b_broadcast =
        std::make_shared<ov::op::v3::Broadcast>(b, ov::op::v0::Constant::create(ngraph::element::i64, {2}, {0, 2}));
    auto bi = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{0, 2});
    {
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{ai, bi}, 0);
        auto body = std::make_shared<ov::Model>(ov::OutputVector{condition, concat}, ov::ParameterVector{ai, bi});
        auto loop = std::make_shared<ov::op::v5::Loop>(trip_count, condition);
        loop->set_special_body_ports({-1, 0});
        loop->set_function(body);
        loop->set_invariant_input(ai, a);
        loop->set_invariant_input(bi, b_broadcast);

        auto loop_res = std::make_shared<ov::op::v0::Result>(loop->get_iter_value(concat));
        auto model = std::make_shared<ov::Model>(ov::OutputVector{loop_res}, ov::ParameterVector{a, b});

        network = CNNNetwork(model);
    }
    {
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{ai}, 0);
        auto body = std::make_shared<ov::Model>(ov::OutputVector{condition, concat}, ov::ParameterVector{ai});
        auto loop = std::make_shared<ov::op::v5::Loop>(trip_count, condition);
        loop->set_special_body_ports({-1, 0});
        loop->set_function(body);
        loop->set_invariant_input(ai, a);

        auto loop_res = std::make_shared<ov::op::v0::Result>(loop->get_iter_value(concat));
        auto model_ref = std::make_shared<ov::Model>(ov::OutputVector{loop_res}, ov::ParameterVector{a, b});

        const auto fc = FunctionsComparator::with_default()
                            .enable(FunctionsComparator::ATTRIBUTES)
                            .enable(FunctionsComparator::CONST_VALUES);
        const auto res = fc.compare(network.getFunction(), model_ref);
        EXPECT_TRUE(res.valid) << res.message;
    }
}

IE_SUPPRESS_DEPRECATED_END
