// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <gmock/gmock.h>

#include <iostream>
#include "common_test_utils/test_assertions.hpp"
#include "compiled_model.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "plugin.hpp"

using ::testing::_;
using ::testing::MatcherCast;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::StrEq;

using namespace ov::mock_autobatch_plugin;

class MockIPlugin : public ov::IPlugin {
public:
    MockIPlugin() {
        set_device_name("HWPLUGIN");
    }
    MOCK_METHOD(std::shared_ptr<ov::ICompiledModel>,
                compile_model,
                (const std::shared_ptr<const ov::Model>&, const ov::AnyMap&),
                (const, override));
    MOCK_METHOD(std::shared_ptr<ov::ICompiledModel>,
                compile_model,
                (const std::shared_ptr<const ov::Model>&, const ov::AnyMap&, const ov::SoPtr<ov::IRemoteContext>&),
                (const, override));
    MOCK_METHOD(ov::Any, get_property, (const std::string&, const ov::AnyMap&), (const, override));
    MOCK_METHOD(void, set_property, (const ov::AnyMap&), (override));
    MOCK_METHOD(ov::SoPtr<ov::IRemoteContext>, create_context, (const ov::AnyMap&), (const, override));
    MOCK_METHOD(ov::SoPtr<ov::IRemoteContext>, get_default_context, (const ov::AnyMap&), (const, override));
    MOCK_METHOD(std::shared_ptr<ov::ICompiledModel>,
                import_model,
                (std::istream&, const ov::AnyMap&),
                (const, override));
    MOCK_METHOD(std::shared_ptr<ov::ICompiledModel>,
                import_model,
                (std::istream&, const ov::SoPtr<ov::IRemoteContext>&, const ov::AnyMap&),
                (const, override));
    MOCK_METHOD(ov::SupportedOpsMap,
                query_model,
                (const std::shared_ptr<const ov::Model>&, const ov::AnyMap&),
                (const, override));
};

class MockAutoBatchInferencePlugin : public Plugin {
public:
    MOCK_METHOD((ov::Any), get_property, (const std::string&, const ov::AnyMap&), (const, override));
};

class MockICompiledModel : public ov::ICompiledModel {
public:
    MockICompiledModel(const std::shared_ptr<const ov::Model>& model, const std::shared_ptr<const ov::IPlugin>& plugin)
        : ov::ICompiledModel(model, plugin) {}
    MOCK_METHOD(std::shared_ptr<ov::ISyncInferRequest>, create_sync_infer_request, (), (const, override));
    MOCK_METHOD(ov::Any, get_property, (const std::string&), (const, override));
    MOCK_METHOD(void, set_property, (const ov::AnyMap&), (override));
    MOCK_METHOD(void, export_model, (std::ostream&), (const, override));
    MOCK_METHOD(std::shared_ptr<const ov::Model>, get_runtime_model, (), (const, override));
    MOCK_METHOD(std::shared_ptr<ov::IAsyncInferRequest>, create_infer_request, (), (const, override));
};

class MockAutoBatchCompileModel : public CompiledModel {
public:
    MockAutoBatchCompileModel(const std::shared_ptr<ov::Model>& model,
                              const std::shared_ptr<const ov::IPlugin>& plugin,
                              const ov::AnyMap& config,
                              const DeviceInformation& device_info,
                              const std::set<std::size_t>& batched_inputs,
                              const std::set<std::size_t>& batched_outputs,
                              const ov::SoPtr<ov::ICompiledModel>& compiled_model_with_batch,
                              const ov::SoPtr<ov::ICompiledModel>& compiled_model_without_batch,
                              const ov::SoPtr<ov::IRemoteContext>& context)
        : CompiledModel(model,
                        plugin,
                        config,
                        device_info,
                        batched_inputs,
                        batched_outputs,
                        compiled_model_with_batch,
                        compiled_model_without_batch,
                        context) {}
    MOCK_METHOD(std::shared_ptr<ov::ISyncInferRequest>, create_sync_infer_request, (), (const, override));
};

class MockISyncInferRequest : public ov::ISyncInferRequest {
public:
    MockISyncInferRequest(const std::shared_ptr<const MockICompiledModel>& compiled_model)
        : ov::ISyncInferRequest(compiled_model) {
        OPENVINO_ASSERT(compiled_model);
        // Allocate input/output tensors
        for (const auto& input : get_inputs()) {
            allocate_tensor(input, [this, input](ov::SoPtr<ov::ITensor>& tensor) {
                // Can add a check to avoid double work in case of shared tensors
                allocate_tensor_impl(tensor,
                                     input.get_element_type(),
                                     input.get_partial_shape().is_dynamic() ? ov::Shape{0} : input.get_shape());
            });
        }
        for (const auto& output : get_outputs()) {
            allocate_tensor(output, [this, output](ov::SoPtr<ov::ITensor>& tensor) {
                // Can add a check to avoid double work in case of shared tensors
                allocate_tensor_impl(tensor,
                                     output.get_element_type(),
                                     output.get_partial_shape().is_dynamic() ? ov::Shape{0} : output.get_shape());
            });
        }
    }
    MOCK_METHOD(std::vector<ov::ProfilingInfo>, get_profiling_info, (), (const, override));
    MOCK_METHOD(void, infer, (), (override));
    MOCK_METHOD(std::vector<ov::SoPtr<ov::IVariableState>>, query_state, (), (const, override));
    ~MockISyncInferRequest() = default;

private:
    void allocate_tensor_impl(ov::SoPtr<ov::ITensor>& tensor,
                              const ov::element::Type& element_type,
                              const ov::Shape& shape) {
        if (!tensor || tensor->get_element_type() != element_type) {
            tensor = ov::make_tensor(element_type, shape);
        } else {
            tensor->set_shape(shape);
        }
    }
};

class MockIAsyncInferRequest : public ov::IAsyncInferRequest {
public:
    MockIAsyncInferRequest(const std::shared_ptr<IInferRequest>& request,
                           const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                           const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor)
        : IAsyncInferRequest(request, task_executor, callback_executor) {
        m_pipeline = {};
    }
    MOCK_METHOD(void, start_async, (), (override));
};