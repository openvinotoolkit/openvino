// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>

#include "ie_icore.hpp"
#include <gmock/gmock.h>
#include <string>
#include <vector>
#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>
#include <cpp_interfaces/interface/ie_iexecutable_network_internal.hpp>

class MockInferencePluginInternal2 : public InferenceEngine::IInferencePlugin {
public:
    MOCK_METHOD(std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>, LoadExeNetworkImpl,
        (const InferenceEngine::CNNNetwork &, (const std::map<std::string, std::string> &)));
    MOCK_METHOD(std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>, LoadNetwork,
        (const InferenceEngine::CNNNetwork &,
         (const std::map<std::string, std::string> &)));
    MOCK_METHOD(void, AddExtension, (InferenceEngine::IExtensionPtr ext_ptr));
    MOCK_METHOD(void, SetConfig, ((const std::map <std::string, std::string> &)));
};

class MockInferencePluginInternal : public InferenceEngine::IInferencePlugin {
public:
    MOCK_METHOD(std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>, LoadExeNetworkImpl,
        (const InferenceEngine::CNNNetwork &, (const std::map<std::string, std::string> &)));
    MOCK_METHOD(void, AddExtension, (InferenceEngine::IExtensionPtr ext_ptr));
    MOCK_METHOD(void, SetConfig, ((const std::map <std::string, std::string> &)));

    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>
    ImportNetwork(std::istream& stream, const std::map <std::string, std::string>&) {
        return {};
    }

    std::string importedString;
};

class MockInferencePluginInternal3 : public InferenceEngine::IInferencePlugin {
public:
    MOCK_METHOD(std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>, LoadExeNetworkImpl,
        (const InferenceEngine::CNNNetwork &, (const std::map<std::string, std::string> &)));
    MOCK_METHOD(void, AddExtension, (InferenceEngine::IExtensionPtr ext_ptr));
    MOCK_METHOD(void, SetConfig, ((const std::map <std::string, std::string> &)));
};
