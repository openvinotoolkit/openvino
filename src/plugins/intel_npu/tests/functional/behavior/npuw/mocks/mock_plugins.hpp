// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <memory>
#include <utility>

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/util/shared_object.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "openvino/core/any.hpp"

//  Based on: openvino/src/inference/tests/functional/caching_test.cpp

namespace ov {
namespace npuw {
namespace tests {

// Need for remote tensor allocation in NPUW JustInferRequest and Weight bank.
// They utilize "create_host_tensor()" method.
// TODO: Mock "create_host_tensor()" method and add tests for it.
class MockRemoteContext : public ov::IRemoteContext {
    std::string m_name;

public:
    MockRemoteContext(std::string name) : m_name(std::move(name)) {}
    const std::string& get_device_name() const override {
        return m_name;
    }
    MOCK_METHOD(ov::SoPtr<ov::IRemoteTensor>,
                create_tensor,
                (const ov::element::Type&, const ov::Shape&, const ov::AnyMap&));
    MOCK_METHOD(const ov::AnyMap&, get_property, (), (const));
};

class MockCompiledModelBase;
using MockCompiledModel = testing::NiceMock<MockCompiledModelBase>;

// Need to also mock async infer request or use infer() call to indicate that start_async() was called.
class MockInferRequestBase : public ov::ISyncInferRequest {
public:
    MockInferRequestBase(const std::shared_ptr<const MockCompiledModel>& compiled_model);
    ~MockInferRequestBase() = default;

    MOCK_METHOD(void, infer, (), (override));
    MOCK_METHOD(std::vector<ov::SoPtr<ov::IVariableState>>, query_state, (), (const, override));
    MOCK_METHOD(std::vector<ov::ProfilingInfo>, get_profiling_info, (), (const, override));

    // This must be called *before* the custom ON_CALL() statements.
    void create_implementation();

private:
    ov::TensorVector create_tensors(std::vector<ov::Output<const ov::Node>> data) {
        ov::TensorVector tensors;
        for (const auto& el : data) {
            const ov::Shape& shape = el.get_shape();
            const ov::element::Type& precision = el.get_element_type();
            tensors.emplace_back(precision, shape);
        }
        return tensors;
    }
};
using MockInferRequest = testing::NiceMock<MockInferRequestBase>;

class MockCompiledModelBase : public ov::ICompiledModel {
public:
    MockCompiledModelBase(
            const std::shared_ptr<const ov::Model>& model, std::shared_ptr<ov::IPlugin> plugin,
            const ov::AnyMap& config,
            std::shared_ptr<std::map<int, std::pair<std::function<void(MockInferRequest&)>, bool>>> infer_reqs_to_expectations);

    // Methods from a base class ov::ICompiledModel
    MOCK_METHOD(const std::vector<ov::Output<const ov::Node>>&, outputs, (), (const, override));
    MOCK_METHOD(const std::vector<ov::Output<const ov::Node>>&, inputs, (), (const, override));
    MOCK_METHOD(std::shared_ptr<ov::IAsyncInferRequest>, create_infer_request, (), (const, override));
    MOCK_METHOD(void, export_model, (std::ostream& model), (const, override));
    MOCK_METHOD(std::shared_ptr<const ov::Model>, get_runtime_model, (), (const, override));
    MOCK_METHOD(void, set_property, (const ov::AnyMap& properties), (override));
    MOCK_METHOD(ov::Any, get_property, (const std::string& name), (const, override));
    MOCK_METHOD(std::shared_ptr<ov::ISyncInferRequest>, create_sync_infer_request, (), (const, override));

    // This must be called *before* the custom ON_CALL() statements.
    void create_implementation();

private:
    std::mutex m_mock_creation_mutex;
    int m_num_created_infer_requests{};
    std::shared_ptr<std::map<int, std::pair<std::function<void(MockInferRequest&)>, bool>>> m_infer_reqs_to_expectations_ptr;

    std::shared_ptr<const ov::Model> m_model;
    ov::AnyMap m_config;
};

template<typename DeviceType>
class MockPluginBase : public ov::IPlugin {
private:
    static constexpr const char* device_name = DeviceType::name;
public:
    MockPluginBase();

    // Normal mock method definitions using gMock.
    // FIXME: There is an issue with mocking of multiple-arguments functions below with generic MOCK_METHOD macro,
    //        on Windows. So, old-style MOCK_METHODn macros are used instead.
    MOCK_CONST_METHOD2_T(compile_model,
                         std::shared_ptr<ov::ICompiledModel>(const std::shared_ptr<const ov::Model>& model,
                                                             const ov::AnyMap& properties));
    MOCK_CONST_METHOD2_T(compile_model,
                         std::shared_ptr<ov::ICompiledModel>(const std::string& model_path,
                                                             const ov::AnyMap& properties));
    MOCK_CONST_METHOD3_T(compile_model,
                         std::shared_ptr<ov::ICompiledModel>(const std::shared_ptr<const ov::Model>& model,
                                                             const ov::AnyMap& properties,
                                                             const ov::SoPtr<ov::IRemoteContext>& context));
    MOCK_CONST_METHOD2_T(get_property, ov::Any(const std::string& name, const ov::AnyMap& arguments));
    MOCK_CONST_METHOD2_T(import_model,
                         std::shared_ptr<ov::ICompiledModel>(std::istream& model,
                                                             const ov::AnyMap& properties));
    MOCK_CONST_METHOD3_T(import_model,
                         std::shared_ptr<ov::ICompiledModel>(std::istream& model,
                                                             const ov::SoPtr<ov::IRemoteContext>& context,
                                                             const ov::AnyMap& properties));
    MOCK_CONST_METHOD2_T(query_model,
                         ov::SupportedOpsMap(const std::shared_ptr<const ov::Model>& model,
                                             const ov::AnyMap& properties));

    MOCK_METHOD(void, set_property, (const ov::AnyMap& properties), (override));
    MOCK_METHOD(ov::SoPtr<ov::IRemoteContext>, create_context, (const ov::AnyMap& remote_properties), (const, override));
    MOCK_METHOD(ov::SoPtr<ov::IRemoteContext>, get_default_context, (const ov::AnyMap& remote_properties), (const, override));

    // This must be called *before* the custom ON_CALL() statements.
    void create_implementation();
    void set_expectations_to_comp_models(int model_idx, std::function<void(MockCompiledModel&)> expectations);
    void set_expectations_to_infer_reqs(int model_idx, int req_idx, std::function<void(MockInferRequest&)> expectations);

    ~MockPluginBase() override;

private:
    std::string m_plugin_name;
    std::mutex m_mock_creation_mutex;  // Internal gmock object registration is not thread-safe
    int m_num_compiled_models{};
    // TODO: Make thread-safe and simplify.
    std::map<int, std::pair<std::function<void(MockCompiledModel&)>, bool>> m_models_to_expectations;
    std::map<int, std::shared_ptr<std::map<int, std::pair<std::function<void(MockInferRequest&)>, bool>>>> m_models_to_reqs_to_expectations;

    // Properties
    int32_t num_streams{0};
    bool exclusive_async_requests = false;
};

namespace mocks {
struct Npu {
    static constexpr const char* name = "MockNPU";
    static constexpr int num_device_ids = 3;
};
struct Cpu {
    static constexpr const char* name = "MockCPU";
    static constexpr int num_device_ids = 2;
};
} //namespace mocks

using MockNpuPlugin = testing::NiceMock<MockPluginBase<mocks::Npu>>;
using MockCpuPlugin = testing::NiceMock<MockPluginBase<mocks::Cpu>>;

}  // namespace tests
}  // namespace npuw
}  // namespace ov
