// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "cpp/ie_cnn_network.h"
#include "xml_father.hpp"
#include "xml_net_builder.hpp"
#include "xml_helper.hpp"
#include "pugixml.hpp"
#include "inference_engine/v2_format_parser.h"
#include <fstream>
#include <stdio.h>
#include "details/ie_exception.hpp"

using namespace std;
using namespace InferenceEngine;
using namespace testing;

class V2TopologyVerificationTests : public ::testing::Test {
protected:
    virtual void TearDown() {}
    virtual void SetUp() {
        xmlHelper.reset(new XMLHelper(new details::V2FormatParser(2)));
    }
public:
    unique_ptr<CNNNetwork> cnnNetwork;
    unique_ptr<XMLHelper> xmlHelper;

    string getNetworkWithConvLayer(string layerPrecision = "Q78", std::vector<size_t > layerInput = { 1, 3, 227, 227 }) {
        std::vector<size_t > inputDims = { 1, 3, 227, 227 };
        std::vector<size_t > outputDims = { 1, 96, 55, 55 };

        return V2NetBuilder::buildNetworkWithOneInput("",layerInput)
            .havingLayers()
                .convolutionLayer(layerPrecision, { {inputDims}, {outputDims} })
            .finish();
    }

    string getNetworkWithConvLayerWithInputPrecision(string inputPrecision, string layerPrecision = "Q78",
                                                     std::vector<size_t > layerInput = {1, 3, 227, 227}) {
        std::vector<size_t > inputDims = {1, 3, 227, 227};
        std::vector<size_t > outputDims = {1, 96, 55, 55};

        return V2NetBuilder::buildNetworkWithOneInput("",layerInput, inputPrecision)
                .havingLayers()
                .convolutionLayer(layerPrecision, {{inputDims}, {outputDims}})
                .finish();
    }

    string getNetworkWithPoolLayer(std::vector<size_t > layerInput = { 1, 3, 227, 227 }) {
        std::vector<size_t > inputDims = { 1, 3, 227, 227 };
        std::vector<size_t > outputDims = { 1, 96, 55, 55 };

        return V2NetBuilder::buildNetworkWithOneInput("",layerInput)
            .havingLayers()
                .poolingLayer({ { inputDims },{ outputDims } })
            .finish();
    }

    string getNetworkWithCropLayer(CropParams params, std::vector<size_t > layerInput = { 1, 3, 227, 227 }) {
        std::vector<size_t > inputDims = { 1, 3, 227, 227 };
        std::vector<size_t > outputDims = { 1, 3, 200, 227 };

        return V2NetBuilder::buildNetworkWithOneInput("",layerInput)
            .havingLayers()
                .cropLayer(params, { {inputDims}, {outputDims} })
            .finish();
    }
};

TEST_F(V2TopologyVerificationTests, testNoThrow) {
    string testContent = getNetworkWithConvLayer();

    xmlHelper->loadContent(testContent);
    try {
        xmlHelper->parse();
    } catch (InferenceEngine::details::InferenceEngineException ex) {
        FAIL() << ex.what();
    }
}

TEST_F(V2TopologyVerificationTests, testDefaultPrecisionsForFP16InputAndOutputLayers) {
    string testContent = getNetworkWithConvLayerWithInputPrecision(Precision(Precision::FP16).name(),
                                                                   Precision(Precision::FP16).name());

    InferenceEngine::details::CNNNetworkImplPtr cnnNetworkImplPtr;
    xmlHelper->loadContent(testContent);
    try {
        cnnNetworkImplPtr = xmlHelper->parseWithReturningNetwork();
    } catch (InferenceEngine::details::InferenceEngineException ex) {
        FAIL() << ex.what();
    }
    OutputsDataMap outputsDataMap;
    cnnNetworkImplPtr->getOutputsInfo(outputsDataMap);
    for (auto outputData: outputsDataMap) {
        ASSERT_TRUE(outputData.second->getPrecision() == Precision::FP32);
    }
    InputsDataMap inputsDataMap;
    cnnNetworkImplPtr->getInputsInfo(inputsDataMap);
    for (auto inputData: inputsDataMap) {
        ASSERT_TRUE(inputData.second->getInputPrecision() == Precision::FP32);
    }
}

TEST_F(V2TopologyVerificationTests, testDefaultPrecisionsFP32InputAndOutputLayers) {
    string testContent = getNetworkWithConvLayerWithInputPrecision(Precision(Precision::FP32).name(),
                                                                   Precision(Precision::FP32).name());

    InferenceEngine::details::CNNNetworkImplPtr cnnNetworkImplPtr;
    xmlHelper->loadContent(testContent);
    try {
        cnnNetworkImplPtr = xmlHelper->parseWithReturningNetwork();
    } catch (InferenceEngine::details::InferenceEngineException ex) {
        FAIL() << ex.what();
    }
    OutputsDataMap outputsDataMap;
    cnnNetworkImplPtr->getOutputsInfo(outputsDataMap);
    for (auto outputData: outputsDataMap) {
        ASSERT_TRUE(outputData.second->getPrecision() == Precision::FP32);
    }
    InputsDataMap inputsDataMap;
    cnnNetworkImplPtr->getInputsInfo(inputsDataMap);
    for (auto inputData: inputsDataMap) {
        ASSERT_TRUE(inputData.second->getInputPrecision() == Precision::FP32);
    }
}

TEST_F(V2TopologyVerificationTests, testDefaultPrecisionsForQ78InputAndOutputLayers) {
    string testContent = getNetworkWithConvLayerWithInputPrecision(Precision(Precision::Q78).name(),
                                                                   Precision(Precision::Q78).name());

    InferenceEngine::details::CNNNetworkImplPtr cnnNetworkImplPtr;
    xmlHelper->loadContent(testContent);
    try {
        cnnNetworkImplPtr = xmlHelper->parseWithReturningNetwork();
    } catch (InferenceEngine::details::InferenceEngineException ex) {
        FAIL() << ex.what();
    }
    OutputsDataMap outputsDataMap;
    cnnNetworkImplPtr->getOutputsInfo(outputsDataMap);
    for (auto outputData: outputsDataMap) {
        ASSERT_TRUE(outputData.second->getPrecision() == Precision::FP32);
    }
    InputsDataMap inputsDataMap;
    cnnNetworkImplPtr->getInputsInfo(inputsDataMap);
    for (auto inputData: inputsDataMap) {
        ASSERT_TRUE(inputData.second->getInputPrecision() == Precision::I16);
    }
}

//convolution input must be 4D
TEST_F(V2TopologyVerificationTests, testCheckConvolutionInputDim_More) {
    string testContent = getNetworkWithConvLayer("Q78", { 1, 1, 3, 227, 227 });

    xmlHelper->loadContent(testContent);
    EXPECT_THROW(xmlHelper->parse(), InferenceEngine::details::InferenceEngineException);
}

//convolution input must be 4D
TEST_F(V2TopologyVerificationTests, testCheckConvolutionInputDim_Less) {
    string testContent = getNetworkWithConvLayer("Q78", { 227, 227 });

    xmlHelper->loadContent(testContent);
    EXPECT_THROW(xmlHelper->parse(), InferenceEngine::details::InferenceEngineException);
}

//pooling input must be 4D
TEST_F(V2TopologyVerificationTests, testCheckPoolingInputDim_Less) {
    string testContent = getNetworkWithPoolLayer({ 227, 227 });
    xmlHelper->loadContent(testContent);
    EXPECT_THROW(xmlHelper->parse(), InferenceEngine::details::InferenceEngineException);
}

//pooling input must be 4D
TEST_F(V2TopologyVerificationTests, testCheckPoolingInputDim_More) {
    string testContent = getNetworkWithPoolLayer({ 1, 1, 3, 227, 227 });
    xmlHelper->loadContent(testContent);
    EXPECT_THROW(xmlHelper->parse(), InferenceEngine::details::InferenceEngineException);
}

TEST_F(V2TopologyVerificationTests, testLeayerPrecisionIsNotMIXED) {
    string testContent = getNetworkWithConvLayer("MIXED");
    xmlHelper->loadContent(testContent);
    EXPECT_THROW(xmlHelper->parse(), InferenceEngine::details::InferenceEngineException);
}

TEST_F(V2TopologyVerificationTests, testMixedPrecisionIfLayerAndNetworkPrecisionsDiffer) {
    string testContent = getNetworkWithConvLayer("Q78");
    xmlHelper->loadContent(testContent);

    try {
        xmlHelper->parse();
    } catch (InferenceEngine::details::InferenceEngineException ex) {
        FAIL() << ex.what();
    }
}

TEST_F(V2TopologyVerificationTests, throwsIfCropDimIsTooBig) {
    CropData data = { 1, 0, 200 };

    string testContent = getNetworkWithCropLayer({ data });
    xmlHelper->loadContent(testContent);
    ASSERT_THROW(xmlHelper->parse(), InferenceEngine::details::InferenceEngineException);
}

TEST_F(V2TopologyVerificationTests, testNoThrowWithProperCropParameters) {
    CropData data = { 2, 0, 200 };

    string testContent = getNetworkWithCropLayer({ data });
    xmlHelper->loadContent(testContent);
    ASSERT_NO_THROW(xmlHelper->parse());
}
