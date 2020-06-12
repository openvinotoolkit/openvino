// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief header file for ICNNNetwork
 * \file ie_icnn_network.hpp
 */
#pragma once

#include <gmock/gmock.h>

#include <memory>
#include <string>

#include "ie_icnn_network.hpp"
#include "cnn_network_impl.hpp"

IE_SUPPRESS_DEPRECATED_START

/**
 * @class ICNNNetwork
 * @brief Main interface to describe the NN topology
 */
class MockICNNNetwork : public InferenceEngine::ICNNNetwork {
 public:
    MOCK_QUALIFIED_METHOD0(getFunction, const noexcept, std::shared_ptr<const ngraph::Function> ());
    MOCK_QUALIFIED_METHOD0(getFunction, noexcept, std::shared_ptr<ngraph::Function>());
    MOCK_QUALIFIED_METHOD0(getPrecision, const noexcept, InferenceEngine::Precision());
    MOCK_QUALIFIED_METHOD1(getOutputsInfo, const noexcept, void(InferenceEngine::OutputsDataMap& out));
    MOCK_QUALIFIED_METHOD1(getInputsInfo, const noexcept, void(InferenceEngine::InputsDataMap &inputs));
    MOCK_QUALIFIED_METHOD1(getInput, const noexcept, InferenceEngine::InputInfo::Ptr(const std::string &inputName));
    MOCK_QUALIFIED_METHOD2(getName, const noexcept, void(char* pName, size_t len));
    MOCK_QUALIFIED_METHOD0(layerCount, const noexcept, size_t());
    MOCK_QUALIFIED_METHOD0(getName, const noexcept, const std::string&());
    MOCK_QUALIFIED_METHOD1(getData, noexcept, InferenceEngine::DataPtr&(const char* dname));
    MOCK_QUALIFIED_METHOD1(addLayer, noexcept, void(const InferenceEngine::CNNLayerPtr& layer));
    MOCK_QUALIFIED_METHOD3(addOutput, noexcept, InferenceEngine::StatusCode(const std::string &, size_t, InferenceEngine::ResponseDesc*));
    MOCK_QUALIFIED_METHOD3(getLayerByName, const noexcept, InferenceEngine::StatusCode(const char* ,
            InferenceEngine::CNNLayerPtr&,
            InferenceEngine::ResponseDesc*));
    MOCK_QUALIFIED_METHOD1(setBatchSize, noexcept, InferenceEngine::StatusCode(const size_t size));
    MOCK_QUALIFIED_METHOD2(setBatchSize, noexcept, InferenceEngine::StatusCode(const size_t size, InferenceEngine::ResponseDesc*));
    MOCK_QUALIFIED_METHOD0(getBatchSize, const noexcept, size_t());
    MOCK_QUALIFIED_METHOD0(Release, noexcept, void());
    MOCK_QUALIFIED_METHOD1(getInputShapes, const noexcept, void(InferenceEngine::ICNNNetwork::InputShapes&));
    MOCK_QUALIFIED_METHOD2(reshape, noexcept, InferenceEngine::StatusCode(const InferenceEngine::ICNNNetwork::InputShapes &, InferenceEngine::ResponseDesc *));
    MOCK_QUALIFIED_METHOD2(AddExtension, noexcept, InferenceEngine::StatusCode(
            const InferenceEngine::IShapeInferExtensionPtr &,
            InferenceEngine::ResponseDesc *));
    MOCK_QUALIFIED_METHOD3(serialize, const noexcept, InferenceEngine::StatusCode(
            const std::string &,
            const std::string &,
            InferenceEngine::ResponseDesc*));
};

/**
 * @class ICNNNetwork
 * @brief Main interface to describe the NN topology
 */
class MockCNNNetworkImpl: public InferenceEngine::details::CNNNetworkImpl {
public:
    MOCK_QUALIFIED_METHOD0(getPrecision, const noexcept, InferenceEngine::Precision());
    MOCK_QUALIFIED_METHOD1(getOutputsInfo, const noexcept, void(InferenceEngine::OutputsDataMap& out));
    MOCK_QUALIFIED_METHOD1(getInputsInfo, const noexcept, void(InferenceEngine::InputsDataMap &inputs));
    MOCK_QUALIFIED_METHOD1(getInput, const noexcept, InferenceEngine::InputInfo::Ptr(const std::string &inputName));
    MOCK_QUALIFIED_METHOD2(getName, const noexcept, void(char* pName, size_t len));
    MOCK_QUALIFIED_METHOD0(getName, const noexcept, const std::string&());
    MOCK_QUALIFIED_METHOD0(layerCount, const noexcept, size_t());
    MOCK_QUALIFIED_METHOD1(getData, noexcept, InferenceEngine::DataPtr&(const char* dname));
    MOCK_QUALIFIED_METHOD1(addLayer, noexcept, void(const InferenceEngine::CNNLayerPtr& layer));
    MOCK_QUALIFIED_METHOD3(addOutput, noexcept, InferenceEngine::StatusCode(const std::string &, size_t , InferenceEngine::ResponseDesc*));
    MOCK_QUALIFIED_METHOD3(getLayerByName, const noexcept, InferenceEngine::StatusCode(
            const char*,
            InferenceEngine::CNNLayerPtr&,
            InferenceEngine::ResponseDesc*));
    MOCK_QUALIFIED_METHOD1(setBatchSize, noexcept, InferenceEngine::StatusCode(const size_t size));
    MOCK_QUALIFIED_METHOD2(setBatchSize, noexcept, InferenceEngine::StatusCode(const size_t size, InferenceEngine::ResponseDesc*));
    MOCK_QUALIFIED_METHOD0(getBatchSize, const noexcept, size_t());
    MOCK_QUALIFIED_METHOD0(Release, noexcept, void());
    MOCK_METHOD1(validate, void(int));

    void validateNetwork() {
        InferenceEngine::details::CNNNetworkImpl::validate();
    }
};

IE_SUPPRESS_DEPRECATED_END
