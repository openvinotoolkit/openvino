// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <gmock/gmock.h>

#include <iostream>

#include "async_infer_request.hpp"
#include "compiled_model.hpp"
#include "ie_icore.hpp"
#include "plugin.hpp"
#include "sync_infer_request.hpp"

namespace MockAutoBatchDevice {

class MockAutoBatchInferencePlugin : public ov::autobatch_plugin::Plugin {
public:
    MOCK_METHOD((ov::autobatch_plugin::DeviceInformation),
                ParseMetaDevices,
                (const std::string&, (const std::map<std::string, std::string>&)),
                (const));
    MOCK_METHOD((ov::autobatch_plugin::DeviceInformation), ParseBatchDevice, (const std::string&), ());

    MOCK_METHOD((InferenceEngine::Parameter),
                GetMetric,
                (const std::string&, (const std::map<std::string, InferenceEngine::Parameter>&)),
                (const, override));
};

class MockAutoBatchExecutableNetwork : public ov::autobatch_plugin::CompiledModel {
public:
    MOCK_METHOD((InferenceEngine::Parameter), GetConfig, (const std::string&), (const, override));
    MOCK_METHOD((InferenceEngine::Parameter), GetMetric, (const std::string&), (const, override));
};

}  // namespace MockAutoBatchDevice
