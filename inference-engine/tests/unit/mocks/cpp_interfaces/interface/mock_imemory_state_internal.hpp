// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_plugin.hpp"
#include "ie_iexecutable_network.hpp"
#include <gmock/gmock.h>
#include <string>
#include <vector>
#include <cpp_interfaces/interface/ie_imemory_state_internal.hpp>

class MockIMemoryStateInternal : public InferenceEngine::IMemoryStateInternal {
 public:
    MOCK_CONST_METHOD0(GetName, std::string ());
    MOCK_METHOD0(Reset, void ());
    MOCK_METHOD1(SetState, void (Blob::Ptr ));
    MOCK_CONST_METHOD0(GetLastState, Blob::CPtr ());
};
