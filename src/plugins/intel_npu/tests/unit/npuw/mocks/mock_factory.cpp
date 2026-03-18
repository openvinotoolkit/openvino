// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "mock_factory.hpp"

namespace ov {
namespace npuw {
namespace tests {

void MockCoreBase::create_implementation() {
    ON_CALL(*this, get_property).WillByDefault([this](const std::string& device_name,
                                                     const std::string& name,
                                                     const AnyMap& arguments) -> ov::Any {
        if (device_name == "NPU" && name == ov::available_devices) {
            return decltype(ov::available_devices)::value_type({"MockNPU"});
        } else if (device_name == "NPU" && name == ov::device::architecture) {
            return decltype(ov::device::architecture)::value_type("5010");
        } else if (device_name == "NPU" && name == ov::intel_npu::compiler_version) {
            return decltype(ov::intel_npu::compiler_version)::value_type(ONEAPI_MAKE_VERSION(7, 29));
        } else {
            OPENVINO_THROW("MockCore doesn't provide property: ", name, " for device: ", device_name);
        }
    });
}

void MockPluginBase::create_implementation() {
    ON_CALL(*this, get_property).WillByDefault([this](const std::string& name, const ov::AnyMap &arguments) -> ov::Any {
        if (name == ov::device::architecture) {
            return decltype(ov::device::architecture)::value_type("5010");
        } else if (name == ov::intel_npu::max_tiles) {
            return decltype(ov::intel_npu::max_tiles)::value_type(4);
        } else if (name == ov::supported_properties) {
            const std::vector<ov::PropertyName> supported_properties = {"NPU_COMPILER_DYNAMIC_QUANTIZATION",
                                                                        ov::num_streams.name(),
                                                                        ov::enable_profiling.name()};
            return decltype(ov::supported_properties)::value_type(supported_properties);
        } else if (name == ov::intel_npu::compiler_version) {
            return decltype(ov::intel_npu::compiler_version)::value_type(ONEAPI_MAKE_VERSION(7, 29));
        } else {
             OPENVINO_THROW("MockPlugin doesn't provide property: ", name);
        }
    });
}

MockNpuwCompiledModelBase::MockNpuwCompiledModelBase(
    const std::shared_ptr<const ov::Model>& model,
    const std::shared_ptr<const ov::IPlugin>& plugin,
    const ov::AnyMap& config)
    : ov::npuw::ICompiledModel(
        std::const_pointer_cast<ov::Model>(const_cast<std::shared_ptr<const ov::Model>&>(model)),
        plugin),
      m_model(model),
      m_config(config) {}

void MockNpuwCompiledModelBase::create_implementation() {
    ON_CALL(*this, inputs()).WillByDefault(testing::ReturnRefOfCopy(m_model->inputs()));
    ON_CALL(*this, outputs()).WillByDefault(testing::ReturnRefOfCopy(m_model->outputs()));
    ON_CALL(*this, create_infer_request()).WillByDefault(testing::Throw(
        std::runtime_error("create_infer_request should not be tested via MockNpuwCompiledModel!"))
    );
    ON_CALL(*this, export_model(testing::_)).WillByDefault([](std::ostream& s) {
        OPENVINO_NOT_IMPLEMENTED;
    });
    ON_CALL(*this, get_runtime_model).WillByDefault(testing::Return(m_model));
    ON_CALL(*this, set_property).WillByDefault([](const ov::AnyMap& properties) {
        OPENVINO_NOT_IMPLEMENTED;
    });
    // TODO: Implement NPUW-specific properties?
    ON_CALL(*this, get_property).WillByDefault(testing::Throw(
        std::runtime_error("get_property() should not be tested via MockNpuwCompiledModel!"))
    );
    ON_CALL(*this, create_sync_infer_request).WillByDefault(testing::Throw(
        std::runtime_error("create_sync_infer_request() should not be tested via MockNpuwCompiledModel!"))
    );
}

void MockNpuwCompiledModelFactoryBase::create_implementation() {
    ON_CALL(*this, create).WillByDefault([this](const std::shared_ptr<ov::Model>& model,
                                                const std::shared_ptr<const ov::IPlugin>& plugin,
                                                const ov::AnyMap& properties) {
            std::lock_guard<std::mutex> lock(m_mock_creation_mutex);

            m_ov_models.push_back(model);
            m_ov_models_properties.push_back(properties);

            auto mock_npuw_compiled_model = std::make_shared<MockNpuwCompiledModel>(model, plugin, properties);
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

MockNpuwCompiledModelFactoryBase::~MockNpuwCompiledModelFactoryBase() {
    for (auto const& idx_to_expectation_and_status : m_models_to_expectations) {
        auto idx = idx_to_expectation_and_status.first;
        auto expectation_and_status = idx_to_expectation_and_status.second;
        if (!expectation_and_status.second) {
            ADD_FAILURE() << "MockNpuwCompiledModelFactory: Expectation for model[" << idx
                          << "] was set, but that model was not created";
        }
    }
}
}  // namespace tests
}  // namespace npuw
}  // namespace ov
