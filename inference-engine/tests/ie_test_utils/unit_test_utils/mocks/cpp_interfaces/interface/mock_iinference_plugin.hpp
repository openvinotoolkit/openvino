// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>

#include <gmock/gmock.h>
#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"

class MockIInferencePlugin : public InferenceEngine::IInferencePlugin {
public:
    MOCK_METHOD1(AddExtension, void(InferenceEngine::IExtensionPtr));
    MOCK_METHOD2(LoadNetwork, std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>(
                const InferenceEngine::CNNNetwork&, const std::map<std::string, std::string>&));
    MOCK_METHOD2(LoadNetwork, std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>(
                const std::string&, const std::map<std::string, std::string>&));
    MOCK_METHOD2(ImportNetwork, std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>(
                const std::string&, const std::map<std::string, std::string>&));
    MOCK_METHOD1(SetConfig, void(const std::map<std::string, std::string> &));

    MOCK_METHOD(void, SetName, (const std::string&), (noexcept));
    MOCK_METHOD(std::string, GetName, (), (const, noexcept));
    MOCK_METHOD(void, SetCore, (InferenceEngine::ICore*), (noexcept));
    MOCK_METHOD(InferenceEngine::ICore *, GetCore, (), (const, noexcept));
    MOCK_CONST_METHOD2(GetConfig, InferenceEngine::Parameter(
                const std::string&, const std::map<std::string, InferenceEngine::Parameter>&));
    MOCK_CONST_METHOD2(GetMetric, InferenceEngine::Parameter(
                const std::string&, const std::map<std::string, InferenceEngine::Parameter>&));
    MOCK_METHOD1(CreateContext,
                InferenceEngine::RemoteContext::Ptr(const InferenceEngine::ParamMap&));
    MOCK_METHOD1(GetDefaultContext, InferenceEngine::RemoteContext::Ptr(const InferenceEngine::ParamMap&));
    MOCK_METHOD3(LoadNetwork, std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>(
                const InferenceEngine::CNNNetwork&, const std::map<std::string, std::string>&,
                InferenceEngine::RemoteContext::Ptr));
    MOCK_METHOD2(ImportNetwork, std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>(
                std::istream&, const std::map<std::string, std::string>&));
    MOCK_METHOD3(ImportNetwork, std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>(
                std::istream&, const InferenceEngine::RemoteContext::Ptr&,
                const std::map<std::string, std::string>&));
    MOCK_CONST_METHOD2(QueryNetwork,
                       InferenceEngine::QueryNetworkResult(const InferenceEngine::CNNNetwork&,
                                                           const std::map<std::string, std::string>&));
};
