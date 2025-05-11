// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock.h>

#include <string>
#include <vector>

#include "openvino/runtime/ivariable_state.hpp"

namespace ov {

class MockIVariableState : public ov::IVariableState {
public:
    MockIVariableState() : ov::IVariableState{"MockIVariableState"} {}
    MOCK_METHOD(const std::string&, get_name, (), (const));
    MOCK_METHOD(void, reset, ());
    MOCK_METHOD(void, set_state, (const ov::SoPtr<ov::ITensor>&));
    MOCK_METHOD(ov::SoPtr<ov::ITensor>, get_state, (), (const));
};

}  // namespace ov
