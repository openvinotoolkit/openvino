// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <gtest/gtest.h>

#include <memory>
#include <random>

#include "test_engine/models/model_builder.hpp"
#include "test_engine/mocks/mock_plugins.hpp"
#include "test_engine/mocks/register_in_ov.hpp"
#include "intel_npu/npuw_private_properties.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/remote_context.hpp"

namespace ov {
namespace weight_sharing {
    struct Context;
}
}
namespace ov {
namespace npuw {
namespace tests {

class RemoteContextAccessor : public ov::RemoteContext {
public:
    static ov::SoPtr<ov::IRemoteContext> as_so_ptr(const ov::RemoteContext& context) {
        RemoteContextAccessor accessor;
        accessor.ov::RemoteContext::operator=(context);
        return ov::SoPtr<ov::IRemoteContext>(accessor._impl, accessor._so);
    }
};

class CrossContextTestsNPUW : public ::testing::Test {
public:
    ov::Core core;
    ov::SoPtr<ov::IRemoteContext> remote_context_gpu;
    ov::SoPtr<ov::IRemoteContext> remote_context_npu;

    std::shared_ptr<ov::Model> model;
    ov::CompiledModel gpu_compiled_model;
    ov::CompiledModel npu_compiled_model;

    size_t model_elem_count = 4096;  // NPU requires a pagesize aligned buffer for shared memory, so 4096 is a good default size.
    void SetUp() override {
        try {
            remote_context_gpu = RemoteContextAccessor::as_so_ptr(core.get_default_context("GPU"));
            remote_context_npu = RemoteContextAccessor::as_so_ptr(core.get_default_context("NPU"));
        } catch (const std::exception& e) {
            GTEST_SKIP() << "Skipping test due to exception while getting remote contexts: " << e.what();
        }
        model = create_shared_context_model(model_elem_count);
        std::tie(gpu_compiled_model, npu_compiled_model) = create_shared_compiled_models(model, remote_context_gpu, remote_context_npu);
    }
protected:
    std::shared_ptr<ov::Model> create_shared_context_model(size_t model_elem_count);
    virtual std::pair<ov::CompiledModel, ov::CompiledModel> create_shared_compiled_models(std::shared_ptr<ov::Model> model,
                                                                                 const ov::SoPtr<ov::IRemoteContext>& remote_context_gpu,
                                                                                 const ov::SoPtr<ov::IRemoteContext>& remote_context_npu);
};

class CrossContextTestsWeightSharingNPUW : public CrossContextTestsNPUW {
public:
    CrossContextTestsWeightSharingNPUW();
    void SetUp() override {
        try {
            remote_context_gpu = RemoteContextAccessor::as_so_ptr(core.get_default_context("GPU"));
            remote_context_npu = RemoteContextAccessor::as_so_ptr(core.get_default_context("NPU"));
        } catch (const std::exception& e) {
            GTEST_SKIP() << "Skipping test due to exception while getting remote contexts: " << e.what();
        }
        model = create_shared_context_model(model_elem_count);
        std::tie(gpu_compiled_model, npu_compiled_model) = create_shared_compiled_models(model, remote_context_gpu, remote_context_npu);
    }
private:
    std::unique_ptr<ov::weight_sharing::Context> m_shared_ctx_ptr;
    std::pair<ov::CompiledModel, ov::CompiledModel> create_shared_compiled_models(std::shared_ptr<ov::Model> model,
                                                                                 const ov::SoPtr<ov::IRemoteContext>& remote_context_gpu, 
                                                                                 const ov::SoPtr<ov::IRemoteContext>& remote_context_npu) override;
};
}  // namespace tests
}  // namespace npuw
}  // namespace ov
