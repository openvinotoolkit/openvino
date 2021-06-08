// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock.h>

#include <string>
#include <memory>
#include <vector>

#include "ie_icnn_network.hpp"

namespace InferenceEngine {

IE_SUPPRESS_DEPRECATED_START

class MockNotEmptyICNNNetwork final : public ICNNNetwork {
public:
    static constexpr const char* INPUT_BLOB_NAME = "first_input";
    const SizeVector INPUT_DIMENTIONS = { 1, 3, 299, 299 };
    static constexpr const char* OUTPUT_BLOB_NAME = "first_output";
    const SizeVector OUTPUT_DIMENTIONS = { 1, 3, 299, 299 };
    const std::string name = "test";
    const std::string& getName() const noexcept override {
        return name;
    }
    void getOutputsInfo(OutputsDataMap& out) const noexcept override;
    void getInputsInfo(InputsDataMap &inputs) const noexcept override;
    std::shared_ptr<ngraph::Function> getFunction() noexcept override {
        return nullptr;
    }
    std::shared_ptr<const ngraph::Function> getFunction() const noexcept override {
        return nullptr;
    }
    MOCK_QUALIFIED_METHOD1(getInput, const noexcept, InputInfo::Ptr(const std::string &inputName));
    MOCK_QUALIFIED_METHOD0(layerCount, const noexcept, size_t());
    MOCK_QUALIFIED_METHOD3(addOutput, noexcept, StatusCode(const std::string &, size_t , ResponseDesc*));
    MOCK_QUALIFIED_METHOD2(setBatchSize, noexcept, StatusCode(const size_t size, ResponseDesc*));
    MOCK_QUALIFIED_METHOD0(getBatchSize, const noexcept, size_t());
    MOCK_QUALIFIED_METHOD1(getInputShapes, const noexcept, void(ICNNNetwork::InputShapes &));
    MOCK_QUALIFIED_METHOD2(reshape, noexcept, StatusCode(const ICNNNetwork::InputShapes &, ResponseDesc *));
    MOCK_QUALIFIED_METHOD3(serialize, const noexcept, StatusCode(const std::string &, const std::string &, InferenceEngine::ResponseDesc*));
};

IE_SUPPRESS_DEPRECATED_END

}  // namespace InferenceEngine
