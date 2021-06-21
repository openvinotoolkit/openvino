// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>

#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include <gmock/gmock.h>

class MockIInferencePlugin : public InferenceEngine::IInferencePlugin {
public:
    MOCK_METHOD(void, AddExtension, (InferenceEngine::IExtensionPtr));
    MOCK_METHOD(std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>, LoadNetwork,
        (const InferenceEngine::CNNNetwork&, (const std::map<std::string, std::string>&)));
    MOCK_METHOD(std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>, LoadNetwork,
        (const std::string&, (const std::map<std::string, std::string>&)));
    MOCK_METHOD(std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>, ImportNetwork,
        (const std::string&, (const std::map<std::string, std::string>&)));
    MOCK_METHOD(void, SetConfig, ((const std::map<std::string, std::string> &)));

    MOCK_METHOD(void, SetName, (const std::string&), (noexcept));
    MOCK_METHOD(std::string, GetName, (), (const, noexcept));
    MOCK_METHOD(void, SetCore, (InferenceEngine::ICore*));
    MOCK_METHOD(InferenceEngine::ICore *, GetCore, (), (const, noexcept));
    MOCK_METHOD(InferenceEngine::Parameter, GetConfig,
        (const std::string&, (const std::map<std::string, InferenceEngine::Parameter>&)));
    MOCK_METHOD(InferenceEngine::Parameter, GetMetric,
        (const std::string&, (const std::map<std::string, InferenceEngine::Parameter>&)));
    MOCK_METHOD(InferenceEngine::RemoteContext::Ptr, CreateContext,
        (const InferenceEngine::ParamMap&));
    MOCK_METHOD(InferenceEngine::RemoteContext::Ptr, GetDefaultContext,
        (const InferenceEngine::ParamMap&));
    MOCK_METHOD(std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>, LoadNetwork,
        (const InferenceEngine::CNNNetwork&, (const std::map<std::string, std::string>&),
         InferenceEngine::RemoteContext::Ptr));
    MOCK_METHOD(std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>, ImportNetwork,
        (std::istream&, (const std::map<std::string, std::string>&)));
    MOCK_METHOD(std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>, ImportNetwork,
        (std::istream&, const InferenceEngine::RemoteContext::Ptr&,
         (const std::map<std::string, std::string>&)));
    MOCK_METHOD(InferenceEngine::QueryNetworkResult, QueryNetwork,
        (const InferenceEngine::CNNNetwork&, (const std::map<std::string, std::string>&)), (const));
};
