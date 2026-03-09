// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "mock_factory.hpp"

namespace ov {
namespace npuw {
namespace tests {
// Need to also mock async infer request or use infer() call to indicate that start_async() was called.
MockNpuwInferRequestBase::MockNpuwInferRequestBase(const std::shared_ptr<const MockNpuwCompiledModel>& compiled_model)
    : ov::ISyncInferRequest(compiled_model) {
    OPENVINO_ASSERT(compiled_model);
}

void MockNpuwInferRequestBase::create_implementation() {
    auto inputs = get_compiled_model()->inputs();
    auto outputs = get_compiled_model()->outputs();
    auto in_tensors = create_tensors(inputs);
    auto out_tensors = create_tensors(outputs);

    for (std::size_t i = 0; i < inputs.size(); ++i) {
        set_tensor(inputs[i], ov::get_tensor_impl(in_tensors[i]));
    }

    for (std::size_t i = 0; i < outputs.size(); ++i) {
        set_tensor(outputs[i], ov::get_tensor_impl(out_tensors[i]));
    }

    ON_CALL(*this, query_state()).WillByDefault([]() -> std::vector<ov::SoPtr<ov::IVariableState>> {
        OPENVINO_NOT_IMPLEMENTED;
    });
    ON_CALL(*this, get_profiling_info()).WillByDefault([]() -> std::vector<ov::ProfilingInfo> {
        OPENVINO_NOT_IMPLEMENTED;
    });
}

MockNpuwCompiledModelBase::MockNpuwCompiledModelBase(
    const std::shared_ptr<const ov::Model>& model,
    const std::shared_ptr<const ov::IPlugin>& plugin,
    const ov::AnyMap& config,
    std::shared_ptr<std::map<int, std::pair<std::function<void(MockNpuwInferRequest&)>, bool>>>
        infer_reqs_to_expectations_ptr)
    : ov::npuw::ICompiledModel(
        std::const_pointer_cast<ov::Model>(const_cast<std::shared_ptr<const ov::Model>&>(model)),
        plugin),
      m_infer_reqs_to_expectations_ptr(infer_reqs_to_expectations_ptr),
      m_model(model),
      m_config(config) {}

void MockNpuwCompiledModelBase::create_implementation() {
    ON_CALL(*this, inputs()).WillByDefault(testing::ReturnRefOfCopy(m_model->inputs()));
    ON_CALL(*this, outputs()).WillByDefault(testing::ReturnRefOfCopy(m_model->outputs()));
    ON_CALL(*this, create_infer_request).WillByDefault([this]() {
        auto syncRequestImpl = create_sync_infer_request();
        return std::make_shared<ov::IAsyncInferRequest>(syncRequestImpl, get_task_executor(), get_callback_executor());
    });
    ON_CALL(*this, export_model(testing::_)).WillByDefault([](std::ostream& s) {
        OPENVINO_NOT_IMPLEMENTED;
    });
    ON_CALL(*this, get_runtime_model).WillByDefault(testing::Return(m_model));
    ON_CALL(*this, set_property).WillByDefault([](const ov::AnyMap& properties) {
        OPENVINO_NOT_IMPLEMENTED;
    });
    // TODO: Implement NPUW-specific properties?
    ON_CALL(*this, get_property).WillByDefault([this](const std::string& name) -> ov::Any {
        if (name == ov::supported_properties) {
            const std::vector<ov::PropertyName> supported_properties = {ov::num_streams.name(),
                                                                        ov::enable_profiling.name()};
            return decltype(ov::supported_properties)::value_type(supported_properties);
        } else if (name == ov::num_streams) {
            if (this->m_config.count(ov::internal::exclusive_async_requests.name())) {
                auto exclusive_async_requests = m_config.at(ov::internal::exclusive_async_requests.name()).as<bool>();
                if (exclusive_async_requests)
                    return ov::streams::Num(1);
            }
            return this->m_config.count(ov::num_streams.name()) ? m_config.at(ov::num_streams.name())
                                                                : ov::streams::Num(1);
        } else if (name == ov::enable_profiling) {
            return this->m_config.count(ov::enable_profiling.name()) ? m_config.at(ov::enable_profiling.name()) : false;
        } else {
            OPENVINO_THROW("get property: " + name);
        }
    });
    ON_CALL(*this, create_sync_infer_request).WillByDefault([this]() {
        std::lock_guard<std::mutex> lock(m_mock_creation_mutex);
        auto mock_npuw_sync_infer_request =
            std::make_shared<MockNpuwInferRequest>(std::dynamic_pointer_cast<const MockNpuwCompiledModel>(shared_from_this()));
        mock_npuw_sync_infer_request->create_implementation();
        if (m_infer_reqs_to_expectations_ptr && m_infer_reqs_to_expectations_ptr->count(m_num_created_infer_requests)) {
            auto& expectation_and_status = (*m_infer_reqs_to_expectations_ptr)[m_num_created_infer_requests];
            auto& expectation = expectation_and_status.first;
            auto& status = expectation_and_status.second;
            expectation(*mock_npuw_sync_infer_request);
            status = true;  // Expectation will be checked
        }

        ++m_num_created_infer_requests;

        return mock_npuw_sync_infer_request;
    });
}

void MockNpuwCompiledModelFactoryBase::create_implementation() {
    ON_CALL(*this, create).WillByDefault([this](const std::shared_ptr<ov::Model>& model,
                                                const std::shared_ptr<const ov::IPlugin>& plugin,
                                                const ov::AnyMap& properties) {
            std::lock_guard<std::mutex> lock(m_mock_creation_mutex);

            m_ov_models.push_back(model);

            std::shared_ptr<std::map<int, std::pair<std::function<void(MockNpuwInferRequest&)>, bool>>>
                infer_reqs_to_expectations;
            if (m_models_to_reqs_to_expectations.count(m_num_compiled_models)) {
                infer_reqs_to_expectations = m_models_to_reqs_to_expectations[m_num_compiled_models];
            }

            auto mock_npuw_compiled_model = std::make_shared<MockNpuwCompiledModel>(model, plugin, properties,
                infer_reqs_to_expectations);

            mock_npuw_compiled_model->create_implementation();

            if (m_models_to_expectations.count(m_num_compiled_models)) {
                auto& expectation_and_status = m_models_to_expectations[m_num_compiled_models];
                auto& expectation = expectation_and_status.first;
                auto& status = expectation_and_status.second;
                expectation(*mock_npuw_compiled_model);
                status = true;  // Expectation will be checked
            }

            ++m_num_compiled_models;

            return mock_npuw_compiled_model;
        });
}

void MockNpuwCompiledModelFactoryBase::set_expectations_to_comp_models(int model_idx,
                                                                   std::function<void(MockNpuwCompiledModel&)> expectations) {
    m_models_to_expectations[model_idx] = std::pair<std::function<void(MockNpuwCompiledModel&)>, bool>(expectations, false);
}

void MockNpuwCompiledModelFactoryBase::set_expectations_to_infer_reqs(int model_idx,
                                                                      int req_idx,
                                                                      std::function<void(MockNpuwInferRequest&)> expectations) {
    if (!m_models_to_reqs_to_expectations.count(model_idx)) {
        m_models_to_reqs_to_expectations[model_idx] =
            std::make_shared<std::map<int, std::pair<std::function<void(MockNpuwInferRequest&)>, bool>>>();
    }
    m_models_to_reqs_to_expectations[model_idx]->insert(
        {req_idx, std::pair<std::function<void(MockNpuwInferRequest&)>, bool>(expectations, false)});
}

MockNpuwCompiledModelFactoryBase::~MockNpuwCompiledModelFactoryBase() {
    for (auto const& idx_to_expectation_and_status : m_models_to_expectations) {
        auto idx = idx_to_expectation_and_status.first;
        auto expectation_and_status = idx_to_expectation_and_status.second;
        if (!expectation_and_status.second) {
            ADD_FAILURE() << "MockNpuwCompiledModelFactory: Expectation for model[" << idx
                          << "] was set, but that model was not compiled";
        }
    }
    for (auto const& idx_to_reqs_to_expectation_and_status : m_models_to_reqs_to_expectations) {
        OPENVINO_ASSERT(idx_to_reqs_to_expectation_and_status.second);
        for (auto const& req_to_expectation_and_status : *idx_to_reqs_to_expectation_and_status.second) {
            auto req_idx = req_to_expectation_and_status.first;
            auto expectation_and_status = req_to_expectation_and_status.second;
            if (!expectation_and_status.second) {
                auto model_idx = idx_to_reqs_to_expectation_and_status.first;
                ADD_FAILURE() << "MockNpuwCompiledModelFactory : Expectation for request[" << req_idx << "] of model["
                              << model_idx << "] was set, but that request was "
                              << "not created";
            }
        }
    }
}
}  // namespace tests
}  // namespace npuw
}  // namespace ov
