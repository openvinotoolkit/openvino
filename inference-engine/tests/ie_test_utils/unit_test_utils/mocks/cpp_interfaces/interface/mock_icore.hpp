// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock.h>
#include "ie_icore.hpp"

class MockICore : public InferenceEngine::ICore {
public:
    MOCK_METHOD((std::shared_ptr<InferenceEngine::ITaskExecutor>), GetTaskExecutor, (), (const));

    MOCK_METHOD(InferenceEngine::CNNNetwork, ReadNetwork,
        (const std::string&, const InferenceEngine::Blob::CPtr&), (const));
    MOCK_METHOD(InferenceEngine::CNNNetwork, ReadNetwork,
        (const std::string&, const std::string&), (const));

    MOCK_METHOD(InferenceEngine::SoExecutableNetworkInternal, LoadNetwork,
        (const InferenceEngine::CNNNetwork&, const std::string&,
        (const std::map<std::string, std::string>&)));
    MOCK_METHOD(InferenceEngine::SoExecutableNetworkInternal, LoadNetwork,
        (const InferenceEngine::CNNNetwork&, const InferenceEngine::RemoteContext::Ptr &,
        (const std::map<std::string, std::string>&)));
    MOCK_METHOD(InferenceEngine::SoExecutableNetworkInternal, LoadNetwork,
        (const std::string &, const std::string &,
        (const std::map<std::string, std::string>&)));

    MOCK_METHOD(InferenceEngine::SoExecutableNetworkInternal, ImportNetwork,
        (std::istream&, const std::string&,
        (const std::map<std::string, std::string>&)));
    MOCK_METHOD(InferenceEngine::SoExecutableNetworkInternal, ImportNetwork,
        (std::istream&, const InferenceEngine::RemoteContext::Ptr&,
        (const std::map<std::string, std::string>&)));

    MOCK_METHOD(InferenceEngine::QueryNetworkResult, QueryNetwork,
        (const InferenceEngine::CNNNetwork&, const std::string&,
        (const std::map<std::string, std::string>&)), (const));

    MOCK_METHOD(InferenceEngine::Parameter, GetMetric, (const std::string&, const std::string&), (const));
    MOCK_METHOD(std::vector<std::string>, GetAvailableDevices, (), (const));
    MOCK_METHOD(bool, DeviceSupportsImportExport, (const std::string&), (const)); // NOLINT not a cast to bool

    ~MockICore() = default;
};
