// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <gmock/gmock.h>
#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>

using namespace InferenceEngine;

class MockExecutableNetworkThreadSafe : public ExecutableNetworkThreadSafeDefault {
public:
    MOCK_METHOD(std::shared_ptr<IInferRequestInternal>, CreateInferRequestImpl,
        (InputsDataMap networkInputs, OutputsDataMap networkOutputs));
    MOCK_METHOD(void, Export, (const std::string &));
    void Export(std::ostream &) override {}
};
