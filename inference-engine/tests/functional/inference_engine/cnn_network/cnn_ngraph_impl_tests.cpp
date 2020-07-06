// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cnn_network_impl.hpp>
#include <details/ie_cnn_network_iterator.hpp>
#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <map>

#include <cpp/ie_cnn_network.h>
#include <ie_util_internal.hpp>
#include <ie_parameter.hpp>
#include <ie_core.hpp>
#include <net_pass.h>

#include <ngraph/opsets/opset3.hpp>
#include <ngraph/function.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/op/maximum.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph/op/convert.hpp>
#include <ngraph/op/parameter.hpp>
#include <ngraph/op/relu.hpp>
#include <ngraph/op/fused/prelu.hpp>
#include <ngraph/op/result.hpp>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "transformations/rt_info/primitives_priority_attribute.hpp"
#include "cnn_network_ngraph_impl.hpp"

using namespace testing;
using namespace InferenceEngine;

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

    InferenceEngine::details::CNNNetworkNGraphImpl cnnNet(ngraph);
    auto convertedNet = std::make_shared<details::CNNNetworkImpl>(cnnNet);
    // Remove convert layer
    InferenceEngine::NetPass::ConvertPrecision(*convertedNet, Precision::I64, Precision::I32);
    ASSERT_NO_THROW(cloneNet(*convertedNet));
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

    InferenceEngine::details::CNNNetworkNGraphImpl cnnNet(ngraph);
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
    getCreatorLayer(cnnNet.getInputsInfo().begin()->second->getInputData());
    cnnNet.addOutput(testLayerName);

    InferenceEngine::OutputsDataMap outs = cnnNet.getOutputsInfo();
    ASSERT_EQ(2, outs.size());
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
    ASSERT_EQ(OK, cnnNet.setBatchSize(2, nullptr));  // triggers conversion
    ASSERT_EQ(2, cnnNet.getBatchSize());
    ASSERT_EQ(nullptr, cnnNet.getFunction());
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
        rtInfo["affinity"] = std::make_shared<ngraph::VariantWrapper<std::string>> (testAffinity);
        relu->set_friendly_name("testReLU");
        auto result = std::make_shared<ngraph::op::Result>(relu);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    InferenceEngine::CNNNetwork cnnNet(ngraph);
    auto cnnLayer = CommonTestUtils::getLayerByName(cnnNet, "testReLU");
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
    getCreatorLayer(cnnNet.getInputsInfo().begin()->second->getInputData());
    auto outputs = cnnNet.getOutputsInfo();
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
    // convert to old representation
    getCreatorLayer(cnnNet.getInputsInfo().begin()->second->getInputData());
    ASSERT_EQ(4, cnnNet.layerCount());
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

    InferenceEngine::details::CNNNetworkNGraphImpl cnnNet(ngraph);
    auto inputInfo = cnnNet.getInput(name);
    ASSERT_EQ(inputInfo->getPreProcess().getResizeAlgorithm(), ResizeAlgorithm::NO_RESIZE);
    inputInfo->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_AREA);
    ASSERT_EQ(inputInfo->getPreProcess().getResizeAlgorithm(), ResizeAlgorithm::RESIZE_AREA);

    auto cnnNetImpl = std::make_shared<details::CNNNetworkImpl>(cnnNet);
    inputInfo = cnnNetImpl->getInput(name);
    ASSERT_EQ(inputInfo->getPreProcess().getResizeAlgorithm(), ResizeAlgorithm::RESIZE_AREA);
}

TEST(CNNNGraphImplTests, SaveAttributesAfterConversion) {
    std::string name = "prelu";
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        auto constant = ngraph::op::Constant::create(ngraph::element::Type_t::f32, {1}, {2});
        auto prelu = std::make_shared<ngraph::op::PRelu>(param, constant);
        prelu->set_friendly_name(name);
        auto add = std::make_shared<ngraph::op::v1::Maximum>(prelu, constant);
        auto result = std::make_shared<ngraph::op::Result>(add);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    InferenceEngine::details::CNNNetworkNGraphImpl cnnNet(ngraph);
    auto * icnnnetwork = static_cast<InferenceEngine::ICNNNetwork*>(&cnnNet);
    CNNLayerPtr layer = CommonTestUtils::getLayerByName(icnnnetwork, name);
    layer->params["test"] = "2";
    layer = CommonTestUtils::getLayerByName(icnnnetwork, name);
    ASSERT_TRUE(layer->params.find("test") != layer->params.end());
    ASSERT_EQ(layer->params["test"], "2");

    // conversion is already triggered, exception is thrown since
    // the ngraph::Function is obsolete
    ASSERT_THROW(std::make_shared<details::CNNNetworkImpl>(cnnNet),
        InferenceEngine::details::InferenceEngineException);
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
    auto cnnLayer = getCreatorLayer(inputInfo.begin()->second->getInputData()).lock();
    ASSERT_TRUE(cnnLayer);
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

        ASSERT_EQ(inputsInfo.at("input")->getPrecision(), Precision::FP32)
                << "FP32 is default presision";
    }
    {
        SCOPED_TRACE("Manually set input precision");

        const auto inputsInfo = cnnNet.getInputsInfo();

        inputsInfo.at("input")->setPrecision(Precision::FP16);
    }
    {
        SCOPED_TRACE("Convert to old format");

        // convert to old representation
        getCreatorLayer(cnnNet.getInputsInfo().begin()->second->getInputData());
    }
    {
        SCOPED_TRACE("After conversion");

        const auto inputsInfo = cnnNet.getInputsInfo();

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

        ASSERT_EQ(inputsInfo.at("input")->getLayout(), Layout::NCHW)
                << "NCHW is default layout";
    }
    {
        SCOPED_TRACE("Manually set input layout");

        const auto inputsInfo = cnnNet.getInputsInfo();

        inputsInfo.at("input")->setLayout(Layout::NHWC);
    }
    {
        SCOPED_TRACE("Convert to old format");

        // convert to old representation
        getCreatorLayer(cnnNet.getInputsInfo().begin()->second->getInputData());
    }
    {
        SCOPED_TRACE("After conversion");

        const auto inputsInfo = cnnNet.getInputsInfo();

        ASSERT_EQ(inputsInfo.at("input")->getLayout(), Layout::NHWC)
                << "Manually set layout should be left unchanged";
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

        ASSERT_EQ(outputsInfo.at("output")->getPrecision(), Precision::FP32)
                << "FP32 is default presision";
    }
    {
        SCOPED_TRACE("Manually set output precision");

        const auto outputsInfo = cnnNet.getOutputsInfo();

        outputsInfo.at("output")->setPrecision(Precision::FP16);
    }
    {
        SCOPED_TRACE("Convert to old format");

        // convert to old representation
        getCreatorLayer(cnnNet.getInputsInfo().begin()->second->getInputData());
    }
    {
        SCOPED_TRACE("After conversion");

        const auto outputsInfo = cnnNet.getOutputsInfo();

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

        ASSERT_EQ(outputsInfo.at("output")->getLayout(), Layout::NCHW)
                << "NCHW is default layout";
    }
    {
        SCOPED_TRACE("Manually set output layout");

        const auto outputsInfo = cnnNet.getOutputsInfo();

        outputsInfo.at("output")->setLayout(Layout::NHWC);
    }
    {
        SCOPED_TRACE("Convert to old format");

        // convert to old representation
        getCreatorLayer(cnnNet.getInputsInfo().begin()->second->getInputData());
    }
    {
        SCOPED_TRACE("After conversion");

        const auto outputsInfo = cnnNet.getOutputsInfo();

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
        auto constant = std::make_shared<ngraph::opset3::Constant>(ngraph::element::f32, ngraph::Shape{1, 2},
                std::vector<float>{1, 2});

        auto read_value = std::make_shared<ngraph::opset3::ReadValue>(constant, "variable_id");
        auto add = std::make_shared<ngraph::opset3::Add>(input, read_value);
        auto result = std::make_shared<ngraph::op::Result>(add);

        ngraph::ParameterVector params = {input};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    InferenceEngine::details::CNNNetworkNGraphImpl cnnNet(ngraph);
    auto convertedNet = std::make_shared<details::CNNNetworkImpl>(cnnNet);
    auto status = convertedNet->setBatchSize(4, nullptr);
    EXPECT_EQ(status, StatusCode::OK);
}

IE_SUPPRESS_DEPRECATED_END
