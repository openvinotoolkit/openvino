// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#pragma once

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "npuw/compiled_model.hpp" // For ov::npuw::ICompiledModel
#include "npuw/llm_compiled_model.hpp" // For ::ov::intel_npu::npuw::llm::INPUWCompiledModelFactory


namespace ov {
namespace npuw {
namespace tests {

class MockCoreBase : public ov::ICore {
public:
    MOCK_METHOD(std::shared_ptr<ov::Model>, read_model, (const std::string& model,
                                                        const ov::Tensor& weights,
                                                        bool frontend_mode), (const, override)); 
    MOCK_METHOD(std::shared_ptr<ov::Model>, read_model, (const std::shared_ptr<AlignedBuffer>& model,
                                                         const std::shared_ptr<AlignedBuffer>& weights), (const, override));
    MOCK_METHOD(std::shared_ptr<ov::Model>, read_model, (const std::filesystem::path& model_path,
                                                         const std::filesystem::path& bin_path,
                                                         const ov::AnyMap& properties), (const, override));
    MOCK_METHOD(ov::AnyMap, create_compile_config, (const std::string& device_name, const ov::AnyMap& origConfig), (const, override));
    MOCK_METHOD(ov::SoPtr<ov::ICompiledModel>, compile_model, (const std::shared_ptr<const ov::Model>& model,
                                                               const std::string& device_name,
                                                               const ov::AnyMap& config), (const, override));
    MOCK_METHOD(ov::SoPtr<ov::ICompiledModel>, compile_model, (const std::shared_ptr<const ov::Model>& model,
                                                               const ov::SoPtr<ov::IRemoteContext>& context,
                                                               const ov::AnyMap& config), (const, override));
    MOCK_METHOD(ov::SoPtr<ov::ICompiledModel>, compile_model, (const std::filesystem::path& model_path,
                                                               const std::string& device_name,
                                                               const ov::AnyMap& config), (const, override));
    MOCK_METHOD(ov::SoPtr<ov::ICompiledModel>, compile_model, (const std::string& model_str,
                                                               const ov::Tensor& weights,
                                                               const std::string& device_name,
                                                               const ov::AnyMap& config), (const, override));
    MOCK_METHOD(ov::SoPtr<ov::ICompiledModel>, import_model, (std::istream& model,
                                                              const std::string& device_name,
                                                              const ov::AnyMap& config), (const, override));
    MOCK_METHOD(ov::SoPtr<ov::ICompiledModel>, import_model, (std::istream& modelStream,
                                                              const ov::SoPtr<ov::IRemoteContext>& context,
                                                              const ov::AnyMap& config), (const, override));
    MOCK_METHOD(ov::SoPtr<ov::ICompiledModel>, import_model, (const ov::Tensor& compiled_blob,
                                                              const std::string& device_name,
                                                              const ov::AnyMap& config), (const, override));
    MOCK_METHOD(ov::SoPtr<ov::ICompiledModel>, import_model, (const ov::Tensor& compiled_blob,
                                                              const ov::SoPtr<ov::IRemoteContext>& context,
                                                              const ov::AnyMap& config), (const, override));
    MOCK_METHOD(ov::SupportedOpsMap, query_model, (const std::shared_ptr<const ov::Model>& model,
                                                   const std::string& device_name,
                                                   const ov::AnyMap& config), (const, override));
    MOCK_METHOD(std::vector<std::string>, get_available_devices, (), (const, override));
    MOCK_METHOD(ov::SoPtr<ov::IRemoteContext>, create_context, (const std::string& device_name, const AnyMap& args), (const, override));
    MOCK_METHOD(ov::SoPtr<ov::IRemoteContext>, get_default_context, (const std::string& device_name), (const, override));
    MOCK_METHOD(Any, get_property, (const std::string& device_name,
                                    const std::string& name,
                                    const AnyMap& arguments), (const, override));
    MOCK_METHOD(AnyMap, get_supported_property, (const std::string& full_device_name,
                                                 const AnyMap& properties,
                                                 const bool keep_core_property), (const, override));
    MOCK_METHOD(bool, device_supports_model_caching, (const std::string& device_name), (const, override));

    // This must be called *before* the custom ON_CALL() statements.
    void create_implementation();

private:
    MOCK_METHOD(void, set_property, (const std::string& device_name, const AnyMap& properties), (override));
};
using MockCore = testing::NiceMock<MockCoreBase>;

class MockPluginBase : public ov::IPlugin {
public:
    MockPluginBase() {}

    // Normal mock method definitions using gMock.
    // FIXME: There is an issue with mocking of multiple-arguments functions below with generic MOCK_METHOD macro,
    //        on Windows. So, old-style MOCK_METHODn macros are used instead.
    MOCK_METHOD(std::shared_ptr<ov::ICompiledModel>, compile_model, (const std::shared_ptr<const ov::Model>& model,
                                                                     const ov::AnyMap& properties), (const, override));
    MOCK_METHOD(std::shared_ptr<ov::ICompiledModel>, compile_model, (const std::filesystem::path& model_path,
                                                                     const ov::AnyMap& properties), (const, override));
    MOCK_METHOD(std::shared_ptr<ov::ICompiledModel>, compile_model, (const std::shared_ptr<const ov::Model>& model,
                                                                     const ov::AnyMap& properties,
                                                                     const ov::SoPtr<ov::IRemoteContext>& context), (const, override));
    MOCK_METHOD(ov::Any, get_property, (const std::string& name, const ov::AnyMap& arguments), (const, override));
    MOCK_METHOD(std::shared_ptr<ov::ICompiledModel>, import_model, (std::istream& model,
                                                                    const ov::AnyMap& properties), (const, override));
    MOCK_METHOD(std::shared_ptr<ov::ICompiledModel>, import_model, (std::istream& model,
                                                                    const ov::SoPtr<ov::IRemoteContext>& context,
                                                                    const ov::AnyMap& properties), (const, override));
    MOCK_METHOD(std::shared_ptr<ov::ICompiledModel>, import_model, (const ov::Tensor& model,
                                                                    const ov::AnyMap& properties), (const, override));
    MOCK_METHOD(std::shared_ptr<ov::ICompiledModel>, import_model, (const ov::Tensor& model,
                                                                    const ov::SoPtr<ov::IRemoteContext>& context,
                                                                    const ov::AnyMap& properties), (const, override));
    MOCK_METHOD(ov::SupportedOpsMap, query_model, (const std::shared_ptr<const ov::Model>& model,
                                                   const ov::AnyMap& properties), (const, override));

    MOCK_METHOD(void, set_property, (const ov::AnyMap& properties), (override));
    MOCK_METHOD(ov::SoPtr<ov::IRemoteContext>, create_context, (const ov::AnyMap& remote_properties), (const, override));
    MOCK_METHOD(ov::SoPtr<ov::IRemoteContext>, get_default_context, (const ov::AnyMap& remote_properties), (const, override));

    // This must be called *before* the custom ON_CALL() statements.
    void create_implementation();
};
using MockPlugin = testing::NiceMock<MockPluginBase>;

class MockNpuwCompiledModelBase : public ov::npuw::ICompiledModel {
public:
    MockNpuwCompiledModelBase(
            const std::shared_ptr<const ov::Model>& model,
            const std::shared_ptr<const ov::IPlugin>& plugin,
            const ov::AnyMap& config);

    // Methods from a base class ov::ICompiledModel
    MOCK_METHOD(const std::vector<ov::Output<const ov::Node>>&, outputs, (), (const, override));
    MOCK_METHOD(const std::vector<ov::Output<const ov::Node>>&, inputs, (), (const, override));
    MOCK_METHOD(std::shared_ptr<ov::IAsyncInferRequest>, create_infer_request, (), (const, override));
    MOCK_METHOD(void, export_model, (std::ostream& model), (const, override));
    MOCK_METHOD(std::shared_ptr<const ov::Model>, get_runtime_model, (), (const, override));
    MOCK_METHOD(void, set_property, (const ov::AnyMap& properties), (override));
    MOCK_METHOD(ov::Any, get_property, (const std::string& name), (const, override));
    MOCK_METHOD(std::shared_ptr<ov::ISyncInferRequest>, create_sync_infer_request, (), (const, override));
    MOCK_METHOD(void, serialize, (std::ostream& stream, const ov::npuw::s11n::CompiledContext& ctx), (const, override));
    MOCK_METHOD(std::shared_ptr<ov::npuw::IBaseInferRequest>, create_base_infer_request, (), (const, override));
    MOCK_METHOD(std::shared_ptr<ov::IAsyncInferRequest>, wrap_async_infer_request,
                (std::shared_ptr<ov::npuw::IBaseInferRequest> internal_request), (const, override));
    MOCK_METHOD(std::string, submodel_device, (const std::size_t idx), (const, override));
    MOCK_METHOD(std::size_t, num_compiled_submodels, (), (const, override));
    MOCK_METHOD(std::shared_ptr<ov::npuw::weights::Bank>, get_weights_bank, (), (const, override));
    MOCK_METHOD(void, set_weights_bank, (std::shared_ptr<ov::npuw::weights::Bank> bank), (override));
    MOCK_METHOD(void, finalize_weights_bank, (), (override));
    MOCK_METHOD(void, reconstruct_closure, (), (override));

    // This must be called *before* the custom ON_CALL() statements.
    void create_implementation();

private:
    std::shared_ptr<const ov::Model> m_model;
    ov::AnyMap m_config;
};
using MockNpuwCompiledModel = testing::NiceMock<MockNpuwCompiledModelBase>;

class MockNpuwCompiledModelFactoryBase : public ov::npuw::INPUWCompiledModelFactory {
public:
    // Generate model is requested to be created first in LLMCompiledModel:
    static constexpr std::size_t kPREFILL_MODEL_INDEX = 1;
    static constexpr std::size_t kGENERATE_MODEL_INDEX = 0;
    static constexpr std::size_t kLM_HEAD_MODEL_INDEX = 2;

    // FIXME: There is an issue with mocking of multiple-arguments functions below with generic MOCK_METHOD macro,
    //        on Windows. So, old-style MOCK_METHODn macros are used instead.
    MOCK_METHOD(std::shared_ptr<ov::npuw::ICompiledModel>, create, (const std::shared_ptr<ov::Model>& model,
                                                                    const std::shared_ptr<const ov::IPlugin>& plugin,
                                                                    const ov::AnyMap& properties), (override));
    MOCK_METHOD(std::shared_ptr<ov::npuw::ICompiledModel>, deserialize, (std::istream& stream,
                                                                         const std::shared_ptr<const ov::IPlugin>& plugin,
                                                                         const ov::AnyMap& properties,
                                                                         const ov::npuw::s11n::CompiledContext& enc_ctx), (override));
    // This must be called *before* the custom ON_CALL() statements.
    void create_implementation();

    void set_expectations_to_comp_models(int model_idx, std::function<void(MockNpuwCompiledModel&)> expectations);

    ~MockNpuwCompiledModelFactoryBase();
public:
    std::mutex m_mock_creation_mutex; // Internal gmock object registration is not thread-safe
    std::vector<std::shared_ptr<ov::Model>> m_ov_models;
    std::vector<ov::AnyMap> m_ov_models_properties;
    int m_num_compiled_models{};
    std::map<int, std::pair<std::function<void(MockNpuwCompiledModel&)>, bool>> m_models_to_expectations;
};

using MockNpuwCompiledModelFactory = testing::NiceMock<MockNpuwCompiledModelFactoryBase>;
}  // namespace tests
}  // namespace npuw
}  // namespace ov
