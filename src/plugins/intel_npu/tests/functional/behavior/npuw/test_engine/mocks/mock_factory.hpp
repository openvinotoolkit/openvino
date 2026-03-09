// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#pragma once

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "intel_npu/config/npuw.hpp"
#include "npuw/compiled_model.hpp" // For ov::npuw::ICompiledModel
#include "npuw/llm_compiled_model.hpp" // For ::ov::intel_npu::npuw::llm::INPUWCompiledModelFactory


namespace ov {
namespace npuw {
namespace tests {

class MockNpuwCompiledModelBase;
using MockNpuwCompiledModel = testing::NiceMock<MockNpuwCompiledModelBase>;

// Need to also mock async infer request or use infer() call to indicate that start_async() was called.
class MockNpuwInferRequestBase : public ov::ISyncInferRequest {
public:
    MockNpuwInferRequestBase(const std::shared_ptr<const MockNpuwCompiledModel>& compiled_model);
    ~MockNpuwInferRequestBase() = default;

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
using MockNpuwInferRequest = testing::NiceMock<MockNpuwInferRequestBase>;

class MockNpuwCompiledModelBase : public ov::npuw::ICompiledModel {
public:
    // NB: static create() method of base class should never be called in tests,
    //     as implementation for it isn't provided in npuw/compiled_model.hpp
    //     header, and we don't want to pull .cpp dependency.
    static std::shared_ptr<ov::npuw::ICompiledModel> create(const std::shared_ptr<ov::Model>& model,
                                                            const std::shared_ptr<const ov::IPlugin>& plugin,
                                                            const ov::AnyMap& properties)
                                                            { OPENVINO_THROW("Internal error: this method should not be called in tests!"); }
    MockNpuwCompiledModelBase(
            const std::shared_ptr<const ov::Model>& model,
            const std::shared_ptr<const ov::IPlugin>& plugin,
            const ov::AnyMap& config,
            std::shared_ptr<std::map<int, std::pair<std::function<void(MockNpuwInferRequest&)>, bool>>> infer_reqs_to_expectations);

    // Methods from a base class ov::ICompiledModel
    MOCK_METHOD(const std::vector<ov::Output<const ov::Node>>&, outputs, (), (const, override));
    MOCK_METHOD(const std::vector<ov::Output<const ov::Node>>&, inputs, (), (const, override));
    MOCK_METHOD(std::shared_ptr<ov::IAsyncInferRequest>, create_infer_request, (), (const, override));
    MOCK_METHOD(void, export_model, (std::ostream& model), (const, override));
    MOCK_METHOD(std::shared_ptr<const ov::Model>, get_runtime_model, (), (const, override));
    MOCK_METHOD(void, set_property, (const ov::AnyMap& properties), (override));
    MOCK_METHOD(ov::Any, get_property, (const std::string& name), (const, override));
    MOCK_METHOD(std::shared_ptr<ov::ISyncInferRequest>, create_sync_infer_request, (), (const, override));

    // Methods from ov::npuw::ICompiledModel interface
    // FIXME: There is an issue with mocking of multiple-arguments functions below with generic MOCK_METHOD macro,
    //        on Windows. So, old-style MOCK_METHODn macros are used instead.
    MOCK_CONST_METHOD2_T(serialize,
                         void(std::ostream& stream, const ov::npuw::s11n::CompiledContext& ctx));
    MOCK_METHOD(std::shared_ptr<ov::npuw::IBaseInferRequest>, create_base_infer_request, (), (const, override));
    MOCK_METHOD(std::shared_ptr<ov::IAsyncInferRequest>, wrap_async_infer_request,
                (std::shared_ptr<ov::npuw::IBaseInferRequest> internal_request), (const, override));
    MOCK_METHOD(std::string, submodel_device, (const std::size_t idx), (const, override));
    MOCK_METHOD(std::vector<ov::npuw::CompiledModelDesc>, get_compiled_submodels, (), (const, override));
    MOCK_METHOD(std::shared_ptr<ov::npuw::weights::Bank>, get_weights_bank, (), (const, override));
    MOCK_METHOD(void, set_weights_bank, (std::shared_ptr<ov::npuw::weights::Bank> bank), (override));
    MOCK_METHOD(void, finalize_weights_bank, (), (override));
    MOCK_METHOD(void, reconstruct_closure, (), (override));

    // This must be called *before* the custom ON_CALL() statements.
    void create_implementation();

private:
    std::mutex m_mock_creation_mutex;
    int m_num_created_infer_requests{};
    std::shared_ptr<std::map<int, std::pair<std::function<void(MockNpuwInferRequest&)>, bool>>> m_infer_reqs_to_expectations_ptr;

    std::shared_ptr<const ov::Model> m_model;
    ov::AnyMap m_config;
};

class MockNpuwCompiledModelFactoryBase : public ::ov::intel_npu::npuw::llm::INPUWCompiledModelFactory {
public:
    // FIXME: There is an issue with mocking of multiple-arguments functions below with generic MOCK_METHOD macro,
    //        on Windows. So, old-style MOCK_METHODn macros are used instead.
    MOCK_METHOD3_T(create,
                   std::shared_ptr<ov::npuw::ICompiledModel>(const std::shared_ptr<ov::Model>& model,
                                                             const std::shared_ptr<const ov::IPlugin>& plugin,
                                                             const ov::AnyMap& properties));
    MOCK_METHOD4_T(deserialize,
                   std::shared_ptr<ov::npuw::ICompiledModel>(std::istream& stream,
                                                             const std::shared_ptr<const ov::IPlugin>& plugin,
                                                             const ov::AnyMap& properties,
                                                             const ov::npuw::s11n::CompiledContext& enc_ctx));

    // This must be called *before* the custom ON_CALL() statements.
    void create_implementation();

    void set_expectations_to_comp_models(int model_idx, std::function<void(MockNpuwCompiledModel&)> expectations);
    void set_expectations_to_infer_reqs(int model_idx, int req_idx, std::function<void(MockNpuwInferRequest&)> expectations);

    ~MockNpuwCompiledModelFactoryBase();
public:
    std::mutex m_mock_creation_mutex; // Internal gmock object registration is not thread-safe
    std::vector<std::shared_ptr<ov::Model>> m_ov_models;
    int m_num_compiled_models{};
    std::map<int, std::pair<std::function<void(MockNpuwCompiledModel&)>, bool>> m_models_to_expectations;
    std::map<int, std::shared_ptr<std::map<int, std::pair<std::function<void(MockNpuwInferRequest&)>, bool>>>> m_models_to_reqs_to_expectations;
};

using MockNpuwCompiledModelFactory = testing::NiceMock<MockNpuwCompiledModelFactoryBase>;
}  // namespace tests
}  // namespace npuw
}  // namespace ov
