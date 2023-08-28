// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock.h>

#include <chrono>
#include <map>
#include <string>
#include <vector>

#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/profiling_info.hpp"

namespace ov {

class MockIAsyncInferRequest : public ov::IAsyncInferRequest {
public:
    MockIAsyncInferRequest() : ov::IAsyncInferRequest(nullptr, nullptr, nullptr) {}

    MOCK_METHOD(void, start_async, ());
    MOCK_METHOD(void, wait, ());
    MOCK_METHOD(bool, wait_for, (const std::chrono::milliseconds&));
    MOCK_METHOD(void, cancel, ());
    MOCK_METHOD(void, set_callback, (std::function<void(std::exception_ptr)>));
    MOCK_METHOD(void, infer, ());
    MOCK_METHOD(std::vector<ov::ProfilingInfo>, get_profiling_info, (), (const));
    MOCK_METHOD(std::vector<ov::SoPtr<ov::IVariableState>>, query_state, (), (const));
};

}  // namespace ov
