// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock.h>
#include "ie_icore.hpp"

class MockICore : public InferenceEngine::ICore {
public:
    MOCK_CONST_METHOD0(GetTaskExecutor, std::shared_ptr<InferenceEngine::ITaskExecutor>());

    MOCK_CONST_METHOD2(ReadNetwork, InferenceEngine::CNNNetwork(const std::string&, const InferenceEngine::Blob::CPtr&));
    MOCK_CONST_METHOD2(ReadNetwork, InferenceEngine::CNNNetwork(const std::string&, const std::string&));

    MOCK_METHOD3(LoadNetwork, InferenceEngine::SoExecutableNetworkInternal(
        const InferenceEngine::CNNNetwork&, const std::string&, const std::map<std::string, std::string>&));
    MOCK_METHOD3(LoadNetwork, InferenceEngine::SoExecutableNetworkInternal(
        const InferenceEngine::CNNNetwork&, const std::shared_ptr<InferenceEngine::RemoteContext> &, const std::map<std::string, std::string>&));
    MOCK_METHOD3(LoadNetwork, InferenceEngine::SoExecutableNetworkInternal(
        const std::string &, const std::string &, const std::map<std::string, std::string>&));

    MOCK_METHOD3(ImportNetwork, InferenceEngine::SoExecutableNetworkInternal(
        std::istream&, const std::string&, const std::map<std::string, std::string>&));
    MOCK_METHOD3(ImportNetwork, InferenceEngine::SoExecutableNetworkInternal(
        std::istream&, const std::shared_ptr<InferenceEngine::RemoteContext>&, const std::map<std::string, std::string>&));

    MOCK_METHOD2(CreateContext, InferenceEngine::RemoteContext::Ptr(const std::string& deviceName,
            const InferenceEngine::ParamMap& params));

    MOCK_CONST_METHOD3(QueryNetwork, InferenceEngine::QueryNetworkResult(
        const InferenceEngine::CNNNetwork&, const std::string&, const std::map<std::string, std::string>&));

    MOCK_CONST_METHOD3(GetMetric, InferenceEngine::Parameter(const std::string&, const std::string&, const std::map<std::string, InferenceEngine::Parameter>&));
    MOCK_CONST_METHOD2(GetConfig, InferenceEngine::Parameter(const std::string&, const std::string&));
    MOCK_CONST_METHOD0(GetAvailableDevices, std::vector<std::string>());
    MOCK_CONST_METHOD1(DeviceSupportsImportExport, bool(const std::string&)); // NOLINT not a cast to bool
    MOCK_CONST_METHOD0(isNewAPI, bool());

    ~MockICore() = default;
};
