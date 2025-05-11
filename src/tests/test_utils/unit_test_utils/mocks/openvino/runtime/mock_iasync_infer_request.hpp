// Copyright (C) 2018-2025 Intel Corporation
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
    MOCK_METHOD(ov::SoPtr<ov::ITensor>, get_tensor, (const ov::Output<const ov::Node>&), (const));
    MOCK_METHOD(std::vector<ov::SoPtr<ov::ITensor>>, get_tensors, (const ov::Output<const ov::Node>&), (const));
    MOCK_METHOD(void, set_tensor, (const ov::Output<const ov::Node>&, const ov::SoPtr<ov::ITensor>&));
    MOCK_METHOD(void, set_tensors, (const ov::Output<const ov::Node>&, const std::vector<ov::SoPtr<ov::ITensor>>&));
    MOCK_METHOD(const std::vector<ov::Output<const ov::Node>>&, get_inputs, (), (const));
    MOCK_METHOD(const std::vector<ov::Output<const ov::Node>>&, get_outputs, (), (const));
};

}  // namespace ov
