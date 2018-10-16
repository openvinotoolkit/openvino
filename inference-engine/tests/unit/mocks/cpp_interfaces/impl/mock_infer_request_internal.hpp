// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_plugin.hpp"
#include "ie_iexecutable_network.hpp"
#include <gmock/gmock.h>
#include <string>
#include <vector>
#include <cpp_interfaces/impl/ie_infer_request_internal.hpp>

using namespace InferenceEngine;

class MockInferRequestInternal : public InferRequestInternal {
public:
    MockInferRequestInternal(InputsDataMap networkInputs, OutputsDataMap networkOutputs)
            : InferRequestInternal(networkInputs, networkOutputs) {}
    using InferRequestInternal::SetBlob;
    using InferRequestInternal::GetBlob;
    MOCK_METHOD0(InferImpl, void());
    MOCK_CONST_METHOD1(GetPerformanceCounts, void(std::map<std::string, InferenceEngineProfileInfo> &));
};
