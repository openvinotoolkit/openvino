// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock.h>

#include <string>
#include <vector>

#include <cpp_interfaces/interface/ie_ivariable_state_internal.hpp>

class MockIVariableStateInternal : public InferenceEngine::IVariableStateInternal {
public:
    MockIVariableStateInternal() : InferenceEngine::IVariableStateInternal{"MockIVariableStateInternal"} {}

    MOCK_METHOD(std::string, GetName, ());
    MOCK_METHOD(void, Reset, ());
    MOCK_METHOD(void, SetState, (const InferenceEngine::Blob::Ptr&));
    MOCK_METHOD(InferenceEngine::Blob::CPtr, GetState, ());
};
