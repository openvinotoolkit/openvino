// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_plugin.hpp"
#include "ie_ihetero_plugin.hpp"
#include "ie_iexecutable_network.hpp"
#include <gmock/gmock.h>
#include <string>
#include <vector>

class MockPluginImpl {
 public:
    MOCK_METHOD1(LoadNetwork, void(InferenceEngine::ICNNNetwork &));
    MOCK_METHOD3(LoadExeNetwork, void(InferenceEngine::IExecutableNetwork::Ptr &,
                                      InferenceEngine::ICNNNetwork &,
                                      const std::map<std::string, std::string> &));
    MOCK_METHOD2(Infer, void(const InferenceEngine::Blob &, InferenceEngine::Blob&));
    MOCK_METHOD2(InferBlobMap, void(const InferenceEngine::BlobMap &, InferenceEngine::BlobMap&));
    /**
     * @brief Handling gtest overloaded function and comparison for reference
     * @param input
     * @param result
     */
    void Infer(const InferenceEngine::BlobMap & input, InferenceEngine::BlobMap& result) {
        InferBlobMap(input, result);
    }

    void LoadNetwork(InferenceEngine::IExecutableNetwork::Ptr &exeNetwork,
                     InferenceEngine::ICNNNetwork &cnnNetwork,
                     const std::map<std::string, std::string> &config) {
        LoadExeNetwork(exeNetwork, cnnNetwork, config);
    }
    MOCK_METHOD1(GetPerformanceCounts, void(std::map <std::string, InferenceEngine::InferenceEngineProfileInfo> &));
    MOCK_METHOD1(AddExtension, void(InferenceEngine::IExtensionPtr ext_ptr));
    MOCK_METHOD1(SetConfig, void ( const std::map <std::string, std::string> &));
    MOCK_METHOD1(SetLogCallback, void (InferenceEngine::IErrorListener &));
    MOCK_METHOD2(ImportNetwork, InferenceEngine::IExecutableNetwork::Ptr(const std::string &,const std::map<std::string, std::string> &));
    /**
     * @depricated Use the version with config parameter
     */
    void QueryNetwork(const InferenceEngine::ICNNNetwork &network, InferenceEngine::QueryNetworkResult &res) const { }
    void QueryNetwork(const InferenceEngine::ICNNNetwork &network,
                      const std::map<std::string, std::string>& config, InferenceEngine::QueryNetworkResult &res) const { }
};
