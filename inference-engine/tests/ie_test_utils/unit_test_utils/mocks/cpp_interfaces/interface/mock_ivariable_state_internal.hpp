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
    MOCK_CONST_METHOD0(GetName, std::string());
    MOCK_METHOD0(Reset, void());
    MOCK_METHOD1(SetState, void(const InferenceEngine::Blob::Ptr&));
    MOCK_CONST_METHOD0(GetState, InferenceEngine::Blob::CPtr());
};
