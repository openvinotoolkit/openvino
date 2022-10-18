// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock.h>
#include "ie_icore.hpp"

class MockICore : public InferenceEngine::ICore {
public:
    MOCK_CONST_METHOD3(ReadNetwork, InferenceEngine::CNNNetwork(const std::string&, const InferenceEngine::Blob::CPtr&, bool));
    MOCK_CONST_METHOD2(ReadNetwork, InferenceEngine::CNNNetwork(const std::string&, const std::string&));

    MOCK_METHOD3(LoadNetwork, InferenceEngine::SoExecutableNetworkInternal(
        const InferenceEngine::CNNNetwork&, const std::string&, const std::map<std::string, std::string>&));
    MOCK_METHOD3(LoadNetwork, InferenceEngine::SoExecutableNetworkInternal(
        const InferenceEngine::CNNNetwork&, const std::shared_ptr<InferenceEngine::RemoteContext> &, const std::map<std::string, std::string>&));
    MOCK_METHOD4(LoadNetwork, InferenceEngine::SoExecutableNetworkInternal(
            const std::string &,
            const std::string &,
            const std::map<std::string, std::string> &,
            const std::function<void(const InferenceEngine::CNNNetwork&)> &));

    MOCK_METHOD3(ImportNetwork, InferenceEngine::SoExecutableNetworkInternal(
        std::istream&, const std::string&, const std::map<std::string, std::string>&));
    MOCK_METHOD3(ImportNetwork, InferenceEngine::SoExecutableNetworkInternal(
        std::istream&, const std::shared_ptr<InferenceEngine::RemoteContext>&, const std::map<std::string, std::string>&));

    MOCK_METHOD2(CreateContext, InferenceEngine::RemoteContext::Ptr(const std::string& deviceName,
            const InferenceEngine::ParamMap& params));

    MOCK_CONST_METHOD3(QueryNetwork, InferenceEngine::QueryNetworkResult(
        const InferenceEngine::CNNNetwork&, const std::string&, const std::map<std::string, std::string>&));

    MOCK_CONST_METHOD3(GetMetric, ov::Any(const std::string&, const std::string&, const ov::AnyMap&));
    MOCK_CONST_METHOD2(GetConfig, ov::Any(const std::string&, const std::string&));
    MOCK_CONST_METHOD3(get_property, ov::Any(const std::string&, const std::string&, const ov::AnyMap&));
    MOCK_METHOD2(set_property, void(const std::string&, const ov::AnyMap&));
    MOCK_CONST_METHOD0(GetAvailableDevices, std::vector<std::string>());
    MOCK_CONST_METHOD1(DeviceSupportsImportExport, bool(const std::string&)); // NOLINT not a cast to bool
    MOCK_METHOD2(GetSupportedConfig, std::map<std::string, std::string>(const std::string&, const std::map<std::string, std::string>&));
    MOCK_CONST_METHOD0(isNewAPI, bool());
    MOCK_METHOD1(GetDefaultContext, InferenceEngine::RemoteContext::Ptr(const std::string&));

    ~MockICore() = default;
};
