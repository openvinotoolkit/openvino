// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock.h>
#include <string>
#include <vector>

#include "ie_imemory_state.hpp"

using namespace InferenceEngine;

class MockIMemoryState : public InferenceEngine::IMemoryState {
public:
    MOCK_QUALIFIED_METHOD3(GetName, const noexcept, StatusCode(char * , size_t, ResponseDesc *));
    MOCK_QUALIFIED_METHOD1(Reset, noexcept, StatusCode(ResponseDesc *));
    MOCK_QUALIFIED_METHOD2(SetState, noexcept, StatusCode(Blob::Ptr, ResponseDesc *));
    MOCK_QUALIFIED_METHOD2(GetLastState, const noexcept, StatusCode(Blob::CPtr &, ResponseDesc *));
};
