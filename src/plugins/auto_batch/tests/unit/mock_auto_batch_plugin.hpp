// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <gmock/gmock.h>

#include <iostream>

#include "auto_batch.hpp"
#include "ie_icore.hpp"

using namespace MockAutoBatchPlugin;
namespace MockAutoBatchDevice {

class MockAutoBatchInferencePlugin : public AutoBatchInferencePlugin {
public:
    MOCK_METHOD((DeviceInformation),
                ParseMetaDevices,
                (const std::string&, (const std::map<std::string, std::string>&)),
                (const));
    MOCK_METHOD((DeviceInformation), ParseBatchDevice, (const std::string&), ());

    MOCK_METHOD((InferenceEngine::Parameter),
                GetMetric,
                (const std::string&, (const std::map<std::string, InferenceEngine::Parameter>&)),
                (const, override));
};

class MockAutoBatchExecutableNetwork : public AutoBatchExecutableNetwork {
public:
    MOCK_METHOD((InferenceEngine::Parameter), GetConfig, (const std::string&), (const, override));
    MOCK_METHOD((InferenceEngine::Parameter), GetMetric, (const std::string&), (const, override));
};

}  // namespace MockAutoBatchDevice
