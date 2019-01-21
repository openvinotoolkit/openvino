// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>

using namespace InferenceEngine;

class MockExecutableNetworkThreadSafe : public ExecutableNetworkThreadSafeDefault {
public:
    MOCK_METHOD2(CreateInferRequestImpl,
                 std::shared_ptr<InferRequestInternal>(InputsDataMap networkInputs, OutputsDataMap networkOutputs));
    MOCK_METHOD1(Export, void(const std::string &));
    MOCK_METHOD1(GetMappedTopology, void(std::map<std::string, std::vector<PrimitiveInfo::Ptr>> &));
};
