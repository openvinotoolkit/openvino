// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_plugin.hpp"
#include "ie_iexecutable_network.hpp"
#include <gmock/gmock.h>
#include <string>
#include <vector>
#include <cpp_interfaces/impl/ie_executable_network_thread_safe_async_only.hpp>

using namespace InferenceEngine;

class MockExecutableNetworkThreadSafeAsyncOnly : public ExecutableNetworkThreadSafeAsyncOnly {
public:
    MOCK_METHOD2(CreateAsyncInferRequestImpl,
                 AsyncInferRequestInternal::Ptr(InputsDataMap networkInputs, OutputsDataMap networkOutputs));
    MOCK_METHOD1(Export, void(const std::string &));
    MOCK_METHOD1(GetMappedTopology, void(std::map<std::string, std::vector<PrimitiveInfo::Ptr>> &));
};
