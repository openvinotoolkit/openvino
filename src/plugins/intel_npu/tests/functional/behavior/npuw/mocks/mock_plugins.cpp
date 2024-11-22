// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mock_plugins.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <utility>

#include "intel_npu/npu_private_properties.hpp"
#include "intel_npu/npuw_private_properties.hpp"
#include "openvino/core/version.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/make_tensor.hpp"

namespace ov {
namespace npuw {
namespace tests {
// Need to also mock async infer request or use infer() call to indicate that start_async() was called.
MockInferRequestBase::MockInferRequestBase(const std::shared_ptr<const MockCompiledModel>& compiled_model)
    : ov::ISyncInferRequest(compiled_model) {
    OPENVINO_ASSERT(compiled_model);
}

void MockInferRequestBase::create_implementation() {
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

MockCompiledModelBase::MockCompiledModelBase(
    const std::shared_ptr<const ov::Model>& model,
    std::shared_ptr<ov::IPlugin> plugin,
    const ov::AnyMap& config,
    std::shared_ptr<std::map<int, std::pair<std::function<void(MockInferRequest&)>, bool>>>
        infer_reqs_to_expectations_ptr)
    : ov::ICompiledModel(model, plugin),
      m_infer_reqs_to_expectations_ptr(infer_reqs_to_expectations_ptr),
      m_model(model),
      m_config(config) {}

void MockCompiledModelBase::create_implementation() {
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
        auto mock_sync_infer_request =
            std::make_shared<MockInferRequest>(std::dynamic_pointer_cast<const MockCompiledModel>(shared_from_this()));
        mock_sync_infer_request->create_implementation();
        if (m_infer_reqs_to_expectations_ptr && m_infer_reqs_to_expectations_ptr->count(m_num_created_infer_requests)) {
            auto& expectation_and_status = (*m_infer_reqs_to_expectations_ptr)[m_num_created_infer_requests];
            auto& expectation = expectation_and_status.first;
            auto& status = expectation_and_status.second;
            expectation(*mock_sync_infer_request);
            status = true;  // Expectation will be checked
        }

        ++m_num_created_infer_requests;

        return mock_sync_infer_request;
    });
}

template <typename DeviceType>
MockPluginBase<DeviceType>::MockPluginBase() {
    m_plugin_name = std::string("openvino_") + std::string(device_name) + std::string("_plugin");
    const ov::Version version = {CI_BUILD_NUMBER, m_plugin_name.c_str()};
    set_device_name(device_name);
    set_version(version);
}

namespace {
ov::PropertyName RO_property(const std::string& propertyName) {
    return ov::PropertyName(propertyName, ov::PropertyMutability::RO);
};

ov::PropertyName RW_property(const std::string& propertyName) {
    return ov::PropertyName(propertyName, ov::PropertyMutability::RW);
};
}  // anonymous namespace

template <typename DeviceType>
void MockPluginBase<DeviceType>::create_implementation() {
    ON_CALL(*this, compile_model(testing::A<const std::shared_ptr<const ov::Model>&>(), testing::_))
        .WillByDefault([this](const std::shared_ptr<const ov::Model>& model, const ov::AnyMap& properties) {
            std::lock_guard<std::mutex> lock(m_mock_creation_mutex);

            std::shared_ptr<std::map<int, std::pair<std::function<void(MockInferRequest&)>, bool>>>
                infer_reqs_to_expectations;
            if (m_models_to_reqs_to_expectations.count(m_num_compiled_models)) {
                infer_reqs_to_expectations = m_models_to_reqs_to_expectations[m_num_compiled_models];
            }

            auto mock_compiled_model =
                std::make_shared<MockCompiledModel>(model, shared_from_this(), properties, infer_reqs_to_expectations);

            mock_compiled_model->create_implementation();

            if (m_models_to_expectations.count(m_num_compiled_models)) {
                auto& expectation_and_status = m_models_to_expectations[m_num_compiled_models];
                auto& expectation = expectation_and_status.first;
                auto& status = expectation_and_status.second;
                expectation(*mock_compiled_model);
                status = true;  // Expectation will be checked
            }

            ++m_num_compiled_models;

            return mock_compiled_model;
        });
    ON_CALL(*this, compile_model(testing::A<const std::string&>(), testing::_))
        .WillByDefault(
            [](const std::string& model_path, const ov::AnyMap& properties) -> std::shared_ptr<ov::ICompiledModel> {
                OPENVINO_NOT_IMPLEMENTED;
            });
    ON_CALL(*this, compile_model(testing::A<const std::shared_ptr<const ov::Model>&>(), testing::_, testing::_))
        .WillByDefault([](const std::shared_ptr<const ov::Model>& model,
                          const ov::AnyMap& properties,
                          const ov::SoPtr<ov::IRemoteContext>& context) -> std::shared_ptr<ov::ICompiledModel> {
            OPENVINO_NOT_IMPLEMENTED;
        });
    ON_CALL(*this, set_property).WillByDefault([this](const ov::AnyMap& properties) {
        for (const auto& it : properties) {
            if (it.first == ov::num_streams.name())
                num_streams = it.second.as<int32_t>();
            else if (it.first == ov::internal::exclusive_async_requests.name())
                exclusive_async_requests = it.second.as<bool>();
            else if (it.first == ov::device::id.name())
                continue;
            else
                OPENVINO_THROW(get_device_name(), " set config: " + it.first);
        }
    });
    ON_CALL(*this, get_property).WillByDefault([this](const std::string& name, const ov::AnyMap& arguments) -> ov::Any {
        std::vector<std::string> device_ids;
        for (int i = 0; i < DeviceType::num_device_ids; ++i) {
            device_ids.push_back(std::to_string(i));
        };
        const static std::vector<ov::PropertyName> roProperties{RO_property(ov::supported_properties.name()),
                                                                RO_property(ov::available_devices.name()),
                                                                RO_property(ov::device::uuid.name()),
                                                                RO_property(ov::device::capabilities.name())};
        // the whole config is RW before network is loaded.
        const static std::vector<ov::PropertyName> rwProperties{RW_property(ov::num_streams.name())};

        std::string device_id;
        if (arguments.find(ov::device::id.name()) != arguments.end()) {
            device_id = arguments.find(ov::device::id.name())->second.as<std::string>();
        }
        if (name == ov::supported_properties) {
            std::vector<ov::PropertyName> supportedProperties;
            supportedProperties.reserve(roProperties.size() + rwProperties.size());
            supportedProperties.insert(supportedProperties.end(), roProperties.begin(), roProperties.end());
            supportedProperties.insert(supportedProperties.end(), rwProperties.begin(), rwProperties.end());

            return decltype(ov::supported_properties)::value_type(supportedProperties);
        } else if (name == ov::internal::supported_properties) {
            return decltype(ov::internal::supported_properties)::value_type(
                {ov::PropertyName{ov::internal::exclusive_async_requests.name(), ov::PropertyMutability::RW}});
        } else if (name == ov::internal::exclusive_async_requests) {
            return decltype(ov::internal::exclusive_async_requests)::value_type{exclusive_async_requests};
        } else if (name == ov::device::uuid) {
            ov::device::UUID uuid;
            for (size_t i = 0; i < uuid.MAX_UUID_SIZE; i++) {
                for (int j = 0; j < DeviceType::num_device_ids; ++j)
                    if (device_id == device_ids[j])
                        uuid.uuid[i] = static_cast<uint8_t>(i * (j + 1));
            }
            return decltype(ov::device::uuid)::value_type{uuid};
        } else if (name == ov::available_devices) {
            return decltype(ov::available_devices)::value_type(device_ids);
        } else if (name == ov::device::capabilities) {
            std::vector<std::string> capabilities;
            capabilities.push_back(ov::device::capability::EXPORT_IMPORT);
            return decltype(ov::device::capabilities)::value_type(capabilities);
        } else if (name == "SUPPORTED_CONFIG_KEYS") {  // TODO: Remove this key
            std::vector<std::string> configs;
            for (const auto& property : rwProperties) {
                configs.emplace_back(property);
            }
            return configs;
        } else if (name == ov::streams::num.name()) {
            return decltype(ov::streams::num)::value_type{num_streams};
        }
        OPENVINO_THROW("Unsupported property: ", name);
    });
    ON_CALL(*this, create_context)
        .WillByDefault([](const ov::AnyMap& remote_properties) -> ov::SoPtr<ov::IRemoteContext> {
            OPENVINO_NOT_IMPLEMENTED;
        });
    // This method is utilized for remote tensor allocation in NPUW JustInferRequest and Weight bank.
    ON_CALL(*this, get_default_context)
        .WillByDefault([](const ov::AnyMap& remote_properties) -> ov::SoPtr<ov::IRemoteContext> {
            return std::make_shared<MockRemoteContext>(device_name);
        });
    ON_CALL(*this, import_model(testing::_, testing::_))
        .WillByDefault([](std::istream& model, const ov::AnyMap& properties) -> std::shared_ptr<ov::ICompiledModel> {
            OPENVINO_NOT_IMPLEMENTED;
        });
    ON_CALL(*this, import_model(testing::_, testing::_, testing::_))
        .WillByDefault([](std::istream& model,
                          const ov::SoPtr<ov::IRemoteContext>& context,
                          const ov::AnyMap& properties) -> std::shared_ptr<ov::ICompiledModel> {
            OPENVINO_NOT_IMPLEMENTED;
        });
    ON_CALL(*this, query_model)
        .WillByDefault(
            [](const std::shared_ptr<const ov::Model>& model, const ov::AnyMap& properties) -> ov::SupportedOpsMap {
                OPENVINO_NOT_IMPLEMENTED;
            });
}

template <typename DeviceType>
void MockPluginBase<DeviceType>::set_expectations_to_comp_models(int model_idx,
                                                                 std::function<void(MockCompiledModel&)> expectations) {
    m_models_to_expectations[model_idx] = std::pair<std::function<void(MockCompiledModel&)>, bool>(expectations, false);
}

template <typename DeviceType>
void MockPluginBase<DeviceType>::set_expectations_to_infer_reqs(int model_idx,
                                                                int req_idx,
                                                                std::function<void(MockInferRequest&)> expectations) {
    if (!m_models_to_reqs_to_expectations.count(model_idx)) {
        m_models_to_reqs_to_expectations[model_idx] =
            std::make_shared<std::map<int, std::pair<std::function<void(MockInferRequest&)>, bool>>>();
    }
    m_models_to_reqs_to_expectations[model_idx]->insert(
        {req_idx, std::pair<std::function<void(MockInferRequest&)>, bool>(expectations, false)});
}

template <typename DeviceType>
MockPluginBase<DeviceType>::~MockPluginBase() {
    for (auto const& idx_to_expectation_and_status : m_models_to_expectations) {
        auto idx = idx_to_expectation_and_status.first;
        auto expectation_and_status = idx_to_expectation_and_status.second;
        if (!expectation_and_status.second) {
            ADD_FAILURE() << DeviceType::name << ": Expectation for model[" << idx
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
                ADD_FAILURE() << DeviceType::name << ": Expectation for request[" << req_idx << "] of model["
                              << model_idx << "] was set, but that request was "
                              << "not created";
            }
        }
    }
}

template class MockPluginBase<ov::npuw::tests::mocks::Npu>;
template class MockPluginBase<ov::npuw::tests::mocks::Cpu>;

}  // namespace tests
}  // namespace npuw
}  // namespace ov
