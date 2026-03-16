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
#include "npuw/base_sync_infer_request.hpp" // For ov::npuw::IBaseInferRequest

namespace ov {
namespace npuw {
namespace tests {

class MockNpuwCompiledModelBase;
using MockNpuwCompiledModel = testing::NiceMock<MockNpuwCompiledModelBase>;

// FIXME: There is an issue with mocking of multiple-arguments functions below with generic MOCK_METHOD macro,
//        on Windows. So, old-style MOCK_METHODn macros are used instead.
class MockNpuwBaseInferRequestBase : public ov::npuw::IBaseInferRequest {
public:
    explicit MockNpuwBaseInferRequestBase(const std::shared_ptr<const MockNpuwCompiledModel>&);

    MOCK_METHOD(void, infer, (), (override));
    MOCK_METHOD2_T(handle_set_remote_input,
                   void(const ov::Output<const ov::Node>& port,
                        const ov::SoPtr<ov::ITensor>& tensor));

    MOCK_METHOD(std::vector<ov::SoPtr<ov::IVariableState>>, query_state, (), (const, override));
    MOCK_METHOD(std::vector<ov::ProfilingInfo>, get_profiling_info, (), (const, override));

    using sptr = std::shared_ptr<IBaseInferRequest>;
    using Completed = std::function<void(std::exception_ptr)>;

    MOCK_METHOD(void, prepare_for_infer, (), (override));
    MOCK_METHOD(bool, valid_subrequest, (std::size_t idx), (const, override));
    MOCK_METHOD(void, start_subrequest, (std::size_t idx), (override));
    MOCK_METHOD2_T(subscribe_subrequest, void(std::size_t idx, Completed cb));
    MOCK_METHOD2_T(run_subrequest_for_success, void(std::size_t idx, bool& failover));
    MOCK_METHOD(void, complete_subrequest, (std::size_t idx), (override));
    MOCK_METHOD(void, cancel_subrequest, (std::size_t idx), (override));
    MOCK_METHOD(std::size_t, total_subrequests, (), (const, override));
    MOCK_METHOD(bool, supports_async_pipeline, (), (const, override));
 
    MOCK_METHOD(void, update_history_size, (int64_t history_size), (override));
    MOCK_METHOD(int64_t, get_history_size, (), (const, override));

    MOCK_METHOD3_T(create_infer_requests,
                   ov::npuw::IBaseInferRequest::RqPtrs(std::size_t id, size_t nireq, bool* recompiled));
    MOCK_METHOD2_T(ensure_subrequest_is_accurate, void(std::size_t idx, bool& failover));
    MOCK_METHOD(void, update_subrequest_links, (std::size_t idx), (override));

    MOCK_METHOD(TensorPtr, alloc_global_out, (std::size_t out_idx), (const, override));
    MOCK_METHOD2_T(bind_global_params, void(std::size_t idx, RqPtr request));
    MOCK_METHOD2_T(bind_global_results, void(std::size_t idx, RqPtr request));
    MOCK_METHOD(bool, needs_copy, (std::size_t idx), (const, override));
    MOCK_CONST_METHOD2_T(needs_copy, bool(std::size_t idx, std::size_t cidx));

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
using MockNpuwBaseInferRequest = testing::NiceMock<MockNpuwBaseInferRequestBase>;

class MockNpuwCompiledModelBase : public ov::npuw::ICompiledModel {
public:
    MockNpuwCompiledModelBase(
            const std::shared_ptr<const ov::Model>& model,
            const std::shared_ptr<const ov::IPlugin>& plugin,
            const ov::AnyMap& config,
            std::shared_ptr<std::map<int, std::pair<std::function<void(MockNpuwBaseInferRequest&)>, bool>>> infer_reqs_to_expectations);

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
    MOCK_METHOD(std::string, get_name, (), (const, override));
    MOCK_METHOD(bool, compile_for_success, (std::size_t id), (override));
    MOCK_METHOD(bool, is_fallback_possible, (std::size_t id), (const, override));
    MOCK_METHOD(std::shared_ptr<ov::npuw::IBaseInferRequest>, create_base_infer_request, (), (const, override));
    MOCK_METHOD(std::shared_ptr<ov::IAsyncInferRequest>, wrap_async_infer_request,
                (std::shared_ptr<ov::npuw::IBaseInferRequest> internal_request), (const, override));
    MOCK_METHOD(std::string, submodel_device, (const std::size_t idx), (const, override));
    MOCK_METHOD(std::vector<ov::npuw::CompiledModelDesc>, get_compiled_submodels, (), (const, override));
    MOCK_METHOD(std::vector<ov::npuw::ICompiledModel::ToSubmodel>, get_inputs_to_submodels_inputs, (), (const, override));
    MOCK_METHOD(std::vector<ov::npuw::ICompiledModel::ToSubmodel>, get_outputs_to_submodels_outputs, (), (const, override));
    using ParamSubscribers = std::map<std::size_t, std::vector<ov::npuw::ICompiledModel::ToSubmodel>>;
    MOCK_METHOD(ParamSubscribers, get_param_subscribers, (), (const, override));
    MOCK_METHOD(ov::npuw::ICompiledModel::SubmodelInsToPrevOuts, get_submodels_input_to_prev_output, (), (const, override));

    MOCK_CONST_METHOD2_T(is_gather_closure,
                         bool(const std::size_t idx, const std::size_t cidx));
    MOCK_METHOD(bool, unpack_required, (const std::size_t idx), (const, override));
    MOCK_CONST_METHOD2_T(unpack_required,
                         bool(const std::size_t idx, const std::size_t cidx));
    MOCK_CONST_METHOD2_T(serialize,
                         void(std::ostream& stream, const ov::npuw::s11n::CompiledContext& ctx));
    MOCK_METHOD(std::shared_ptr<ov::npuw::weights::Bank>, get_weights_bank, (), (const, override));
    MOCK_METHOD(void, set_weights_bank, (std::shared_ptr<ov::npuw::weights::Bank> bank), (override));
    MOCK_METHOD(void, finalize_weights_bank, (), (override));
    MOCK_METHOD(void, reconstruct_closure, (), (override));
    MOCK_METHOD(std::string, global_mem_device, (), (const, override));
    MOCK_METHOD(std::string, funcall_mem_device, (const std::size_t idx), (const, override));
    MOCK_METHOD(bool, acc_check_enabled, (), (const, override));
    MOCK_METHOD(std::string, get_acc_ref_device, (), (const, override));
    MOCK_CONST_METHOD2_T(is_accurate,
                         bool(const ov::SoPtr<ov::ITensor>& actual, const ov::SoPtr<ov::ITensor>& reference));
    MOCK_METHOD(void, log_device_dist, (), (const, override));

    // This must be called *before* the custom ON_CALL() statements.
    void create_implementation();

private:
    std::mutex m_mock_creation_mutex;
    int m_num_created_infer_requests{};
    std::shared_ptr<std::map<int, std::pair<std::function<void(MockNpuwBaseInferRequest&)>, bool>>> m_infer_reqs_to_expectations_ptr;

    std::shared_ptr<const ov::Model> m_model;
    ov::AnyMap m_config;
};

class MockNpuwCompiledModelFactoryBase : public ::ov::intel_npu::npuw::llm::INPUWCompiledModelFactory {
public:
    static constexpr std::size_t kGENERATE_IDX = 0;
    static constexpr std::size_t kPREFILL_IDX = 1;
    static constexpr std::size_t kLM_HEAD_IDX = 2;

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
    void set_expectations_to_infer_reqs(int model_idx, int req_idx, std::function<void(MockNpuwBaseInferRequest&)> expectations);

    ~MockNpuwCompiledModelFactoryBase();
public:
    std::mutex m_mock_creation_mutex; // Internal gmock object registration is not thread-safe
    std::vector<std::shared_ptr<ov::Model>> m_ov_models;
    int m_num_compiled_models{};
    std::map<int, std::pair<std::function<void(MockNpuwCompiledModel&)>, bool>> m_models_to_expectations;
    std::map<int, std::shared_ptr<std::map<int, std::pair<std::function<void(MockNpuwBaseInferRequest&)>, bool>>>> m_models_to_reqs_to_expectations;
};

using MockNpuwCompiledModelFactory = testing::NiceMock<MockNpuwCompiledModelFactoryBase>;
}  // namespace tests
}  // namespace npuw
}  // namespace ov
