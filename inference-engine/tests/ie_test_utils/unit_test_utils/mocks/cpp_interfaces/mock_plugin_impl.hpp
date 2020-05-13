// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock.h>

#include <map>
#include <string>
#include <vector>

#include "ie_plugin.hpp"
#include "ie_iexecutable_network.hpp"
#include "ie_icore.hpp"
#include "cpp/ie_executable_network.hpp"


class MockPluginImpl {
 public:
    MOCK_METHOD3(LoadExeNetwork, void(InferenceEngine::IExecutableNetwork::Ptr &,
                                      const InferenceEngine::ICNNNetwork &,
                                      const std::map<std::string, std::string> &));

    void LoadNetwork(InferenceEngine::IExecutableNetwork::Ptr &exeNetwork,
                     const InferenceEngine::ICNNNetwork &cnnNetwork,
                     const std::map<std::string, std::string> &config) {
        LoadExeNetwork(exeNetwork, cnnNetwork, config);
    }
    MOCK_METHOD1(AddExtension, void(InferenceEngine::IExtensionPtr ext_ptr));
    MOCK_METHOD1(SetConfig, void(const std::map <std::string, std::string> &));
    IE_SUPPRESS_DEPRECATED_START
    MOCK_METHOD1(SetLogCallback, void(InferenceEngine::IErrorListener &));
    IE_SUPPRESS_DEPRECATED_END
    MOCK_METHOD2(ImportNetwork, InferenceEngine::IExecutableNetwork::Ptr(const std::string &, const std::map<std::string, std::string> &));
    InferenceEngine::ExecutableNetwork ImportNetwork(const std::istream&, const std::map<std::string, std::string> &) {return {};}
    MOCK_QUALIFIED_METHOD0(GetName, const noexcept, std::string(void));
    MOCK_QUALIFIED_METHOD1(SetName, noexcept, void(const std::string &));
    MOCK_QUALIFIED_METHOD0(GetCore, const noexcept, InferenceEngine::ICore*(void));
    MOCK_QUALIFIED_METHOD1(SetCore, noexcept, void(InferenceEngine::ICore*));

    MOCK_CONST_METHOD2(GetConfig, InferenceEngine::Parameter(const std::string& name,
                                                const std::map<std::string, InferenceEngine::Parameter> & options));
    MOCK_CONST_METHOD2(GetMetric, InferenceEngine::Parameter(const std::string& name,
                                                const std::map<std::string, InferenceEngine::Parameter> & options));
    void QueryNetwork(const InferenceEngine::ICNNNetwork &network,
                      const std::map<std::string, std::string>& config, InferenceEngine::QueryNetworkResult &res) const { }

    MOCK_METHOD1(CreateContext, InferenceEngine::RemoteContext::Ptr(const std::map<std::string, InferenceEngine::Parameter> & options));
    MOCK_METHOD0(GetDefaultContext, InferenceEngine::RemoteContext::Ptr(void));
    InferenceEngine::ExecutableNetwork LoadNetwork(const InferenceEngine::ICNNNetwork &, const std::map<std::string, std::string> &,
        InferenceEngine::RemoteContext::Ptr) { return{}; }
    InferenceEngine::ExecutableNetwork ImportNetwork(const std::istream &networkModel,
                                                     const InferenceEngine::RemoteContext::Ptr &context,
                                                     const std::map<std::string, std::string> &config = {}) { return{}; }
};
