// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief header file for MockICNNNetwork
 * \file mock_icnn_network.hpp
 */
#pragma once

#include <gmock/gmock.h>

#include <memory>
#include <string>

#include "ie_icnn_network.hpp"

IE_SUPPRESS_DEPRECATED_START

namespace InferenceEngine {
class CNNLayer;
}  // namespace InferenceEngine

/**
 * @class MockICNNNetwork
 * @brief Main interface to describe the NN topology
 */
class MockICNNNetwork final : public InferenceEngine::ICNNNetwork {
 public:
    MOCK_QUALIFIED_METHOD0(getFunction, const noexcept, std::shared_ptr<const ngraph::Function> ());
    MOCK_QUALIFIED_METHOD0(getFunction, noexcept, std::shared_ptr<ngraph::Function>());
    MOCK_QUALIFIED_METHOD1(getOutputsInfo, const noexcept, void(InferenceEngine::OutputsDataMap& out));
    MOCK_QUALIFIED_METHOD1(getInputsInfo, const noexcept, void(InferenceEngine::InputsDataMap &inputs));
    MOCK_QUALIFIED_METHOD1(getInput, const noexcept, InferenceEngine::InputInfo::Ptr(const std::string &inputName));
    MOCK_QUALIFIED_METHOD0(layerCount, const noexcept, size_t());
    MOCK_QUALIFIED_METHOD0(getName, const noexcept, const std::string&());
    MOCK_QUALIFIED_METHOD3(addOutput, noexcept, InferenceEngine::StatusCode(const std::string &, size_t, InferenceEngine::ResponseDesc*));
     MOCK_QUALIFIED_METHOD3(getLayerByName, const noexcept, InferenceEngine::StatusCode(const char* ,
            std::shared_ptr<InferenceEngine::CNNLayer>&,
            InferenceEngine::ResponseDesc*));
    MOCK_QUALIFIED_METHOD2(setBatchSize, noexcept, InferenceEngine::StatusCode(const size_t size, InferenceEngine::ResponseDesc*));
    MOCK_QUALIFIED_METHOD0(getBatchSize, const noexcept, size_t());
    MOCK_QUALIFIED_METHOD1(getInputShapes, const noexcept, void(InferenceEngine::ICNNNetwork::InputShapes&));
    MOCK_QUALIFIED_METHOD2(reshape, noexcept, InferenceEngine::StatusCode(const InferenceEngine::ICNNNetwork::InputShapes &, InferenceEngine::ResponseDesc *));
    MOCK_QUALIFIED_METHOD3(serialize, const noexcept, InferenceEngine::StatusCode(
            const std::string &,
            const std::string &,
            InferenceEngine::ResponseDesc*));
};
