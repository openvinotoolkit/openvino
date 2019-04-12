// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_iinfer_request.hpp>
#include "mock_executable_network_internal.hpp"
#include <gmock/gmock.h>
#include <string>
#include <vector>
#include <cpp_interfaces/impl/ie_plugin_internal.hpp>
#include <cpp_interfaces/impl/ie_executable_network_internal.hpp>

class MockInferencePluginInternal2 : public InferenceEngine::InferencePluginInternal {
public:
    using InferenceEngine::IInferencePluginInternal::Infer;
    using InferenceEngine::InferencePluginInternal::LoadNetwork;
    MOCK_METHOD2(LoadExeNetworkImpl, std::shared_ptr<InferenceEngine::ExecutableNetworkInternal>(
            InferenceEngine::ICNNNetwork &,const std::map<std::string, std::string> &));
    MOCK_METHOD3(LoadNetwork, void(
            InferenceEngine::IExecutableNetwork::Ptr &,
            InferenceEngine::ICNNNetwork &,
            const std::map<std::string, std::string> &));
    MOCK_METHOD1(AddExtension, void(InferenceEngine::IExtensionPtr ext_ptr));
    MOCK_METHOD1(SetConfig, void ( const std::map <std::string, std::string> &));
};

class MockInferencePluginInternal : public InferenceEngine::InferencePluginInternal {
public:
    using InferenceEngine::IInferencePluginInternal::Infer;
    using InferenceEngine::IInferencePluginInternal::LoadNetwork;
    MOCK_METHOD2(LoadExeNetworkImpl, std::shared_ptr<InferenceEngine::ExecutableNetworkInternal>(
            InferenceEngine::ICNNNetwork &,const std::map<std::string, std::string> &));
    MOCK_METHOD2(Infer, void(const InferenceEngine::BlobMap &, InferenceEngine::BlobMap&));
    MOCK_METHOD1(AddExtension, void(InferenceEngine::IExtensionPtr ext_ptr));
    MOCK_METHOD1(SetConfig, void ( const std::map <std::string, std::string> &));
};

class MockInferencePluginInternal3 : public InferenceEngine::InferencePluginInternal {
public:
    using InferenceEngine::IInferencePluginInternal::Infer;
    using InferenceEngine::IInferencePluginInternal::LoadNetwork;
    MOCK_METHOD2(LoadExeNetworkImpl, std::shared_ptr<InferenceEngine::ExecutableNetworkInternal>(
            InferenceEngine::ICNNNetwork &,const std::map<std::string, std::string> &));
    MOCK_METHOD1(AddExtension, void(InferenceEngine::IExtensionPtr ext_ptr));
    MOCK_METHOD1(SetConfig, void ( const std::map <std::string, std::string> &));
};
