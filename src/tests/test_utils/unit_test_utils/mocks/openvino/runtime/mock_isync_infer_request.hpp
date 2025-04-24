// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock.h>

#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/profiling_info.hpp"

namespace ov {

class MockISyncInferRequest : public ov::ISyncInferRequest {
public:
    MockISyncInferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled_model)
        : ov::ISyncInferRequest(compiled_model) {}
    MOCK_METHOD(void, infer, ());
    MOCK_METHOD(std::vector<ov::ProfilingInfo>, get_profiling_info, (), (const));
    MOCK_METHOD(std::vector<ov::SoPtr<ov::IVariableState>>, query_state, (), (const));
};

}  // namespace ov
