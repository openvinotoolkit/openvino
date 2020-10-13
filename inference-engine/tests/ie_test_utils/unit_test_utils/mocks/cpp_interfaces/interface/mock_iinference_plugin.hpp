// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>

#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include <gmock/gmock.h>

class MockIInferencePlugin : public InferenceEngine::IInferencePlugin {
public:
    MOCK_METHOD1(AddExtension, void(InferenceEngine::IExtensionPtr));
    MOCK_METHOD2(LoadNetwork, InferenceEngine::ExecutableNetwork(
                const ICNNNetwork&, const std::map<std::string, std::string>&));
    MOCK_METHOD2(ImportNetwork, InferenceEngine::ExecutableNetwork(
                const std::string&, const std::map<std::string, std::string>&));
    MOCK_METHOD1(SetConfig, void(const std::map<std::string, std::string> &));

    MOCK_QUALIFIED_METHOD1(SetName, noexcept, void(const std::string&));
    MOCK_QUALIFIED_METHOD0(GetName, const noexcept, std::string(void));
    MOCK_QUALIFIED_METHOD1(SetCore, noexcept, void(InferenceEngine::ICore*));
    MOCK_QUALIFIED_METHOD0(GetCore, const noexcept, InferenceEngine::ICore *(void));
    MOCK_QUALIFIED_METHOD2(GetConfig, const, InferenceEngine::Parameter(
                const std::string&, const std::map<std::string, InferenceEngine::Parameter>&));
    MOCK_QUALIFIED_METHOD2(GetMetric, const, InferenceEngine::Parameter(
                const std::string&, const std::map<std::string, InferenceEngine::Parameter>&));
    MOCK_METHOD1(CreateContext,
                InferenceEngine::RemoteContext::Ptr(const InferenceEngine::ParamMap&));
    MOCK_METHOD0(GetDefaultContext, InferenceEngine::RemoteContext::Ptr(void));
    MOCK_METHOD3(LoadNetwork, InferenceEngine::ExecutableNetwork(
                const InferenceEngine::ICNNNetwork&, const std::map<std::string, std::string>&,
                InferenceEngine::RemoteContext::Ptr));
    MOCK_METHOD2(ImportNetwork, InferenceEngine::ExecutableNetwork(
                std::istream&, const std::map<std::string, std::string>&));
    MOCK_METHOD3(ImportNetwork, InferenceEngine::ExecutableNetwork(
                std::istream&, const InferenceEngine::RemoteContext::Ptr&,
                const std::map<std::string, std::string>&));
};
