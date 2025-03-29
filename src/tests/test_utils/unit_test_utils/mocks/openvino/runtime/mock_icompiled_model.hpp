// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock.h>

#include <map>
#include <string>
#include <vector>

#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/icompiled_model.hpp"

namespace ov {

class MockICompiledModel : public ov::ICompiledModel {
public:
    MockICompiledModel(const std::shared_ptr<const ov::Model>& model, const std::shared_ptr<const ov::IPlugin>& plugin)
        : ov::ICompiledModel(model, plugin) {}
    MOCK_METHOD(const std::vector<ov::Output<const ov::Node>>&, outputs, (), (const));
    MOCK_METHOD(const std::vector<ov::Output<const ov::Node>>&, inputs, (), (const));
    MOCK_METHOD(std::shared_ptr<ov::IAsyncInferRequest>, create_infer_request, (), (const));
    MOCK_METHOD(std::shared_ptr<const ov::Model>, get_runtime_model, (), (const));
    MOCK_METHOD(void, export_model, (std::ostream&), (const));

    MOCK_METHOD(void, set_property, (const ov::AnyMap& config));
    MOCK_METHOD(ov::Any, get_property, (const std::string& name), (const));
    MOCK_METHOD(ov::SoPtr<ov::IRemoteContext>, get_context, (), (const));

    MOCK_CONST_METHOD0(create_sync_infer_request, std::shared_ptr<ov::ISyncInferRequest>(void));

    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request_default() const {
        return create_async_infer_request();
    }
};

}  // namespace ov
