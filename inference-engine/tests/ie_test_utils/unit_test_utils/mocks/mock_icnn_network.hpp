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
    MOCK_METHOD(std::shared_ptr<const ngraph::Function>, getFunction, (), (const, noexcept));
    MOCK_METHOD(std::shared_ptr<ngraph::Function>, getFunction, (), (noexcept));
    MOCK_METHOD(void, getOutputsInfo, (InferenceEngine::OutputsDataMap& out), (const, noexcept));
    MOCK_METHOD(void, getInputsInfo, (InferenceEngine::InputsDataMap &inputs), (const, noexcept));
    MOCK_METHOD(InferenceEngine::InputInfo::Ptr, getInput, (const std::string &inputName), (const, noexcept));
    MOCK_METHOD(size_t, layerCount, (), (const, noexcept));
    MOCK_METHOD(const std::string&, getName, (), (const, noexcept));
    MOCK_METHOD(InferenceEngine::StatusCode, addOutput,
        (const std::string &, size_t, InferenceEngine::ResponseDesc*), (noexcept));
    MOCK_METHOD(InferenceEngine::StatusCode, getLayerByName,
        (const char* , std::shared_ptr<InferenceEngine::CNNLayer>&, InferenceEngine::ResponseDesc*),
        (const, noexcept));
    MOCK_METHOD(InferenceEngine::StatusCode, setBatchSize,
        (const size_t size, InferenceEngine::ResponseDesc*), (noexcept));
    MOCK_METHOD(size_t, getBatchSize, (), (const, noexcept));
    MOCK_METHOD(void, getInputShapes, (InferenceEngine::ICNNNetwork::InputShapes&), (const, noexcept));
    MOCK_METHOD(InferenceEngine::StatusCode, reshape,
        (const InferenceEngine::ICNNNetwork::InputShapes &, InferenceEngine::ResponseDesc *),
        (const, noexcept));
    MOCK_METHOD(InferenceEngine::StatusCode, serialize,
        (const std::string &, const std::string &, InferenceEngine::ResponseDesc*), (const, noexcept));
    MOCK_METHOD(InferenceEngine::StatusCode, serialize,
        (std::ostream &, std::ostream &, InferenceEngine::ResponseDesc*), (const, noexcept));
    MOCK_METHOD(InferenceEngine::StatusCode, serialize,
        (std::ostream &, InferenceEngine::Blob::Ptr &, InferenceEngine::ResponseDesc*), (const, noexcept));
};
