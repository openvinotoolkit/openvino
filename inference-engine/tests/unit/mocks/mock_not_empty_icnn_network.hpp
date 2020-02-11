// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_plugin.hpp"
#include "ie_iexecutable_network.hpp"
#include "ie_icnn_network.hpp"
#include <gmock/gmock.h>
#include <string>
#include <memory>
#include <vector>

namespace InferenceEngine {

class MockNotEmptyICNNNetwork : public ICNNNetwork {
public:
    static constexpr const char* INPUT_BLOB_NAME = "first_input";
    const SizeVector INPUT_DIMENTIONS = { 1, 3, 299, 299 };
    static constexpr const char* OUTPUT_BLOB_NAME = "first_output";
    const SizeVector OUTPUT_DIMENTIONS = { 1, 3, 299, 299 };
    const std::string name = "test";
    Precision getPrecision() const noexcept {
        return Precision::FP32;
    }
    const std::string& getName() const noexcept {
        return name;
    }
    void getOutputsInfo(OutputsDataMap& out) const noexcept override {
        auto data = std::make_shared<Data>(MockNotEmptyICNNNetwork::OUTPUT_BLOB_NAME, Precision::UNSPECIFIED);
        data->getInputTo()[""] = std::make_shared<CNNLayer>(LayerParams{
            MockNotEmptyICNNNetwork::OUTPUT_BLOB_NAME,
            "FullyConnected",
            Precision::FP32 });
        out[MockNotEmptyICNNNetwork::OUTPUT_BLOB_NAME] = data;
    };
    void getInputsInfo(InputsDataMap &inputs) const noexcept override {
        auto inputInfo = std::make_shared<InputInfo>();

        auto inData = std::make_shared<Data>(MockNotEmptyICNNNetwork::INPUT_BLOB_NAME, Precision::UNSPECIFIED);
        auto inputLayer = std::make_shared<CNNLayer>(LayerParams{
            MockNotEmptyICNNNetwork::INPUT_BLOB_NAME,
            "Input",
            Precision::FP32 });
        inData->getInputTo()[MockNotEmptyICNNNetwork::OUTPUT_BLOB_NAME] = inputLayer;
        inData->setDims(MockNotEmptyICNNNetwork::INPUT_DIMENTIONS);
        inData->setLayout(Layout::NCHW);
        inputInfo->setInputData(inData);

        auto outData = std::make_shared<Data>(MockNotEmptyICNNNetwork::OUTPUT_BLOB_NAME, Precision::UNSPECIFIED);
        outData->setDims(MockNotEmptyICNNNetwork::OUTPUT_DIMENTIONS);
        outData->setLayout(Layout::NCHW);
        outData->getInputTo()[""] = std::make_shared<CNNLayer>(LayerParams{
            MockNotEmptyICNNNetwork::OUTPUT_BLOB_NAME,
            "FullyConnected",
            Precision::FP32 });

        inputLayer->outData.push_back(outData);

        inputs[MockNotEmptyICNNNetwork::INPUT_BLOB_NAME] = inputInfo;
    };
    void addLayer(const CNNLayerPtr& layer) noexcept override {}
    const std::shared_ptr<const ngraph::Function> getFunction() const noexcept override {
        return nullptr;
    }
    MOCK_QUALIFIED_METHOD1(getInput, const noexcept, InputInfo::Ptr (const std::string &inputName));
    MOCK_QUALIFIED_METHOD2(getName, const noexcept, void (char* pName, size_t len));
    MOCK_QUALIFIED_METHOD0(layerCount, const noexcept, size_t ());
    MOCK_QUALIFIED_METHOD1(getData, noexcept, DataPtr&(const char* dname));
    MOCK_QUALIFIED_METHOD3(addOutput, noexcept, StatusCode (const std::string &, size_t , ResponseDesc*));
    MOCK_QUALIFIED_METHOD3(getLayerByName, const noexcept, StatusCode (const char* , CNNLayerPtr& , ResponseDesc* ));
    MOCK_QUALIFIED_METHOD1(setBatchSize, noexcept, StatusCode (const size_t size));
    MOCK_QUALIFIED_METHOD2(setBatchSize, noexcept, StatusCode (const size_t size, ResponseDesc*));
    MOCK_QUALIFIED_METHOD0(getBatchSize, const noexcept, size_t ());
    MOCK_QUALIFIED_METHOD0(Release, noexcept, void ());
    MOCK_QUALIFIED_METHOD1(getInputShapes, const noexcept, void (ICNNNetwork::InputShapes &));
    MOCK_QUALIFIED_METHOD2(reshape, noexcept, StatusCode (const ICNNNetwork::InputShapes &, ResponseDesc *));
    MOCK_QUALIFIED_METHOD2(AddExtension, noexcept, StatusCode (const IShapeInferExtensionPtr &, ResponseDesc *));
    MOCK_QUALIFIED_METHOD3(serialize, const noexcept, StatusCode (const std::string &, const std::string &, InferenceEngine::ResponseDesc*));
};

}  // namespace InferenceEngine
