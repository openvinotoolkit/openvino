// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fake_plugins.hpp"

#include <memory>
#include <string>

#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/shared_object.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"

using namespace ov::npuw::tests;
namespace {

bool support_model(const std::shared_ptr<const ov::Model>& model, const ov::SupportedOpsMap& supported_ops) {
    for (const auto& op : model->get_ops()) {
        if (supported_ops.find(op->get_friendly_name()) == supported_ops.end())
            return false;
    }
    return true;
}

ov::PropertyName RO_property(const std::string& propertyName) {
    return ov::PropertyName(propertyName, ov::PropertyMutability::RO);
};

ov::PropertyName RW_property(const std::string& propertyName) {
    return ov::PropertyName(propertyName, ov::PropertyMutability::RW);
};

}  // namespace

// Mock plugins

FakeCompiledModel::FakeCompiledModel(const std::shared_ptr<const ov::Model>& model,
                                     const std::shared_ptr<const ov::IPlugin>& plugin,
                                     const ov::AnyMap& config)
    : ov::ICompiledModel(model, plugin), m_config(config), m_model(model) {
}

// Methods from a base class ov::ICompiledModel
void FakeCompiledModel::export_model(std::ostream& model) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<const ov::Model> FakeCompiledModel::get_runtime_model() const {
    auto model = m_model->clone();
    // Add execution information into the model
    size_t exec_order = 0;
    for (const auto& op : model->get_ordered_ops()) {
        auto& info = op->get_rt_info();
        info[ov::exec_model_info::EXECUTION_ORDER] = std::to_string(exec_order++);
        info[ov::exec_model_info::IMPL_TYPE] = get_plugin()->get_device_name() + "_ " + op->get_type_info().name;
        auto perf_count_enabled = get_property(ov::enable_profiling.name()).as<bool>();
        info[ov::exec_model_info::PERF_COUNTER] = perf_count_enabled ? "0" : "not_executed";
        std::string original_names = ov::getFusedNames(op);
        if (original_names.empty()) {
            original_names = op->get_friendly_name();
        } else if (original_names.find(op->get_friendly_name()) == std::string::npos) {
            original_names = op->get_friendly_name() + "," + original_names;
        }
        info[ov::exec_model_info::ORIGINAL_NAMES] = original_names;
        if (op->inputs().size() > 0)
            info[ov::exec_model_info::RUNTIME_PRECISION] = op->get_input_element_type(0);
        else
            info[ov::exec_model_info::RUNTIME_PRECISION] = op->get_output_element_type(0);

        std::stringstream precisions_ss;
        for (size_t i = 0; i < op->get_output_size(); i++) {
            if (i > 0)
                precisions_ss << ",";
            precisions_ss << op->get_output_element_type(i);
        }
        info[ov::exec_model_info::OUTPUT_PRECISIONS] = precisions_ss.str();
    }
    return model;
}

void FakeCompiledModel::set_property(const ov::AnyMap& properties) {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::Any FakeCompiledModel::get_property(const std::string& name) const {
    if (name == ov::supported_properties) {
        const std::vector<ov::PropertyName> supported_properties = {ov::num_streams.name(),
                                                                    ov::enable_profiling.name()};
        return decltype(ov::supported_properties)::value_type(supported_properties);
    } else if (name == ov::num_streams) {
        if (m_config.count(ov::internal::exclusive_async_requests.name())) {
            auto exclusive_async_requests = m_config.at(ov::internal::exclusive_async_requests.name()).as<bool>();
            if (exclusive_async_requests)
                return ov::streams::Num(1);
        }
        return m_config.count(ov::num_streams.name()) ? m_config.at(ov::num_streams.name()) : ov::streams::Num(1);
    } else if (name == ov::enable_profiling) {
        return m_config.count(ov::enable_profiling.name()) ? m_config.at(ov::enable_profiling.name()) : false;
    } else {
        OPENVINO_THROW("get property: " + name);
    }
}

const std::shared_ptr<const ov::Model>& FakeCompiledModel::get_model() const {
    return m_model;
}

ov::SoPtr<ov::IRemoteContext> FakeCompiledModel::get_context() const {
    OPENVINO_NOT_IMPLEMENTED;
}

FakeInferRequest::FakeInferRequest(const std::shared_ptr<const FakeCompiledModel>& compiled_model)
        : ov::ISyncInferRequest(compiled_model) {
    OPENVINO_ASSERT(compiled_model);
    m_model = compiled_model->get_model();
    // Allocate input/output tensors
    for (const auto& input : get_inputs()) {
        allocate_tensor(input, [this, input, compiled_model](ov::SoPtr<ov::ITensor>& tensor) {
            // Can add a check to avoid double work in case of shared tensors
            allocate_tensor_impl(tensor, input.get_element_type(),
                                 input.get_partial_shape().is_dynamic() ? ov::Shape{0} : input.get_shape());
        });
    }
    for (const auto& output : get_outputs()) {
        allocate_tensor(output, [this, output, compiled_model](ov::SoPtr<ov::ITensor>& tensor) {
            // Can add a check to avoid double work in case of shared tensors
            allocate_tensor_impl(tensor, output.get_element_type(),
                                 output.get_partial_shape().is_dynamic() ? ov::Shape{0} : output.get_shape());
        });
    }
}

void FakeInferRequest::infer() {
    ov::TensorVector input_tensors;
    for (const auto& input : get_inputs()) {
        input_tensors.emplace_back(ov::make_tensor(get_tensor(input)));
    }
    ov::TensorVector output_tensors;
    for (const auto& output : get_outputs()) {
        output_tensors.emplace_back(ov::make_tensor(get_tensor(output)));
    }
    m_model->evaluate(output_tensors, input_tensors);
}

std::vector<ov::SoPtr<ov::IVariableState>> FakeInferRequest::query_state() const {
     OPENVINO_NOT_IMPLEMENTED;
}
std::vector<ov::ProfilingInfo> FakeInferRequest::get_profiling_info() const {
    OPENVINO_NOT_IMPLEMENTED;
}

void FakeInferRequest::allocate_tensor_impl(ov::SoPtr<ov::ITensor>& tensor,
                          const ov::element::Type& element_type,
                          const ov::Shape& shape) {
    if (!tensor || tensor->get_element_type() != element_type) {
        tensor = ov::SoPtr<ov::ITensor>(ov::make_tensor(element_type, shape), nullptr);
    } else {
        tensor->set_shape(shape);
    }
}

std::shared_ptr<ov::ISyncInferRequest> FakeCompiledModel::create_sync_infer_request() const {
    return std::make_shared<FakeInferRequest>(std::dynamic_pointer_cast<const FakeCompiledModel>(shared_from_this()));
}

FakePluginBase::FakePluginBase(const std::string& name,
                               const std::unordered_set<std::string>& supported_ops,
                               bool dynamism_supported)
        : m_supported_ops(supported_ops),
          m_dynamism_supported(dynamism_supported) {
        set_device_name(name);
    }

std::shared_ptr<ov::ICompiledModel> FakePluginBase::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                                  const ov::AnyMap& properties) const {
    OPENVINO_ASSERT(model);
    if (!support_model(model, query_model(model, properties)))
        OPENVINO_THROW("Unsupported model");
    return std::make_shared<FakeCompiledModel>(model, shared_from_this(), properties);
}

std::shared_ptr<ov::ICompiledModel> FakePluginBase::compile_model(const std::string& model_path,
                                                                  const ov::AnyMap& properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::ICompiledModel> FakePluginBase::compile_model(
        const std::shared_ptr<const ov::Model>& model,
        const ov::AnyMap& properties,
        const ov::SoPtr<ov::IRemoteContext>& context) const {
    OPENVINO_NOT_IMPLEMENTED;
}

void FakePluginBase::set_property(const ov::AnyMap& properties) {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::Any FakePluginBase::get_property(const std::string& name, const ov::AnyMap& arguments) const {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::SoPtr<ov::IRemoteContext> FakePluginBase::create_context(const ov::AnyMap& remote_properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::SoPtr<ov::IRemoteContext> FakePluginBase::get_default_context(const ov::AnyMap& remote_properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::ICompiledModel> FakePluginBase::import_model(std::istream& model,
                                                                 const ov::AnyMap& properties) const {
    std::string xmlString, xmlInOutString;
    ov::Tensor weights;

    ov::pass::StreamSerialize::DataHeader hdr = {};
    model.read(reinterpret_cast<char*>(&hdr), sizeof hdr);

    model.seekg(hdr.custom_data_offset);
    xmlInOutString.resize(hdr.custom_data_size);
    model.read(const_cast<char*>(xmlInOutString.c_str()), hdr.custom_data_size);

    // read blob content
    model.seekg(hdr.consts_offset);
    if (hdr.consts_size) {
        weights = ov::Tensor(ov::element::i8, ov::Shape{hdr.consts_size});
        char* data = static_cast<char*>(weights.data());
        model.read(data, hdr.consts_size);
    }

    // read XML content
    model.seekg(hdr.model_offset);
    xmlString.resize(hdr.model_size);
    model.read(const_cast<char*>(xmlString.c_str()), hdr.model_size);

    ov::Core core;
    auto ov_model = core.read_model(xmlString, weights);
    return compile_model(ov_model, properties);
}

std::shared_ptr<ov::ICompiledModel> FakePluginBase::import_model(std::istream& model,
                                                               const ov::SoPtr<ov::IRemoteContext>& context,
                                                               const ov::AnyMap& properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::SupportedOpsMap FakePluginBase::query_model(const std::shared_ptr<const ov::Model>& model,
                                                const ov::AnyMap& properties) const {
    OPENVINO_ASSERT(model);
    ov::SupportedOpsMap res;
    auto device_id = properties.count(ov::device::id.name())
                         ? properties.at(ov::device::id.name()).as<std::string>()
                         : m_default_device_id;

    auto supported = ov::get_supported_nodes(
        model,
        [&](std::shared_ptr<ov::Model>& model) {
            ov::pass::Manager manager;
            manager.register_pass<ov::pass::InitNodeInfo>();
            manager.register_pass<ov::pass::ConstantFolding>();
            manager.run_passes(model);
        },
        [&](const std::shared_ptr<ov::Node>& op) {
            if (op->is_dynamic() && !m_dynamism_supported)
                return false;
            if (m_supported_ops.find(op->get_type_info().name) == m_supported_ops.end())
                return false;
            return true;
        });
    for (auto&& op_name : supported) {
        res.emplace(op_name, get_device_name() + "." + device_id);
    }
    return res;
}

FakeNpuPlugin::FakeNpuPlugin(const std::string& name)
        : FakePluginBase(name, {"Parameter", "Result", "Add", "Constant", "Reshape"}, true) {
    set_version(get_const_version());
}

const ov::Version& FakeNpuPlugin::get_const_version()  {
    static const ov::Version version = {CI_BUILD_NUMBER, "openvino_fake_npu_plugin"};
    return version;
}
void FakeNpuPlugin::set_property(const ov::AnyMap& properties) {
    for (const auto& it : properties) {
        if (it.first == ov::num_streams.name())
            num_streams = it.second.as<int32_t>();
        else if (it.first == ov::enable_profiling.name())
            m_profiling = it.second.as<bool>();
        else if (it.first == ov::internal::exclusive_async_requests.name())
            exclusive_async_requests = it.second.as<bool>();
        else if (it.first == ov::device::id.name())
            continue;
        else
            OPENVINO_THROW(get_device_name(), " set config: " + it.first);
    }
}

ov::Any FakeNpuPlugin::get_property(const std::string& name, const ov::AnyMap& arguments) const  {
    const static std::vector<std::string> device_ids = {"0", "1", "2"};
    const static std::vector<ov::PropertyName> roProperties{
        RO_property(ov::supported_properties.name()),
        RO_property(ov::available_devices.name()),
        RO_property(ov::loaded_from_cache.name()),
        RO_property(ov::device::uuid.name()),
      //  RO_property(METRIC_KEY(IMPORT_EXPORT_SUPPORT)),
    };
    // the whole config is RW before network is loaded.
    const static std::vector<ov::PropertyName> rwProperties{
        RW_property(ov::num_streams.name()),
        RW_property(ov::enable_profiling.name()),
    };

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
            {ov::PropertyName{ov::internal::caching_properties.name(), ov::PropertyMutability::RO},
             ov::PropertyName{ov::internal::exclusive_async_requests.name(), ov::PropertyMutability::RW}});
    } else if (name == ov::internal::exclusive_async_requests) {
        return decltype(ov::internal::exclusive_async_requests)::value_type{exclusive_async_requests};
    } else if (name == ov::device::uuid) {
        ov::device::UUID uuid;
        for (size_t i = 0; i < uuid.MAX_UUID_SIZE; i++) {
            if (device_id == device_ids[0])
                uuid.uuid[i] = static_cast<uint8_t>(i);
            else if (device_id == device_ids[1])
                uuid.uuid[i] = static_cast<uint8_t>(i * 2);
            else if (device_id == device_ids[2])
                uuid.uuid[i] = static_cast<uint8_t>(i * 3);
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
//        } else if (METRIC_KEY(IMPORT_EXPORT_SUPPORT) == name) {
//            return true;
    } else if (ov::internal::caching_properties == name) {
        std::vector<ov::PropertyName> caching_properties = {ov::device::uuid};
        return decltype(ov::internal::caching_properties)::value_type(caching_properties);
    } else if (name == "SUPPORTED_METRICS") {  // TODO: Remove this key
        std::vector<std::string> configs;
        for (const auto& property : roProperties) {
            configs.emplace_back(property);
        }
        return configs;
    } else if (name == ov::loaded_from_cache.name()) {
        return m_loaded_from_cache;
    } else if (name == ov::enable_profiling.name()) {
        return decltype(ov::enable_profiling)::value_type{m_profiling};
    } else if (name == ov::streams::num.name()) {
        return decltype(ov::streams::num)::value_type{num_streams};
    }
    OPENVINO_THROW("Unsupported property: ", name);
}

FakeCpuPlugin::FakeCpuPlugin(const std::string& name)
        : FakePluginBase(name, {"Parameter", "Result", "Add", "Constant", "Subtract", "Reshape"}) {
    set_version(get_const_version());
}

const ov::Version& FakeCpuPlugin::get_const_version()  {
    static const ov::Version version = {CI_BUILD_NUMBER, "openvino_fake_cpu_plugin"};
    return version;
}

void FakeCpuPlugin::set_property(const ov::AnyMap& properties)  {
    for (const auto& it : properties) {
        if (it.first == ov::enable_profiling.name())
            m_profiling = it.second.as<bool>();
        else if (it.first == ov::device::id.name())
            continue;
        else
            OPENVINO_THROW(get_device_name(), " set config: " + it.first);
    }
}

ov::Any FakeCpuPlugin::get_property(const std::string& name, const ov::AnyMap& arguments) const  {
    const static std::vector<std::string> device_ids = {"0", "1"};
    const static std::vector<ov::PropertyName> roProperties{
        RO_property(ov::supported_properties.name()),
        RO_property(ov::available_devices.name()),
        RO_property(ov::loaded_from_cache.name()),
        RO_property(ov::device::uuid.name()),
        //RO_property(METRIC_KEY(IMPORT_EXPORT_SUPPORT)),
    };
    // the whole config is RW before network is loaded.
    const static std::vector<ov::PropertyName> rwProperties{
        RW_property(ov::num_streams.name()),
        RW_property(ov::enable_profiling.name()),
    };
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
            {ov::PropertyName{ov::internal::caching_properties.name(), ov::PropertyMutability::RO}});
    } else if (name == ov::device::uuid) {
        ov::device::UUID uuid;
        for (size_t i = 0; i < uuid.MAX_UUID_SIZE; i++) {
            if (device_id == device_ids[0])
                uuid.uuid[i] = static_cast<uint8_t>(i * 2);
            else if (device_id == device_ids[1])
                uuid.uuid[i] = static_cast<uint8_t>(i * 4);
        }
        return decltype(ov::device::uuid)::value_type{uuid};
    } else if (name == ov::available_devices) {
        return decltype(ov::available_devices)::value_type(device_ids);
    } else if (name == ov::device::capabilities) {
        std::vector<std::string> capabilities;
        capabilities.push_back(ov::device::capability::EXPORT_IMPORT);
        return decltype(ov::device::capabilities)::value_type(capabilities);
    } else if (name == ov::loaded_from_cache.name()) {
        return m_loaded_from_cache;
    } else if (name == ov::enable_profiling.name()) {
        return decltype(ov::enable_profiling)::value_type{m_profiling};
    } else if (name == "SUPPORTED_CONFIG_KEYS") {  // TODO: Remove this key
        std::vector<std::string> configs;
        for (const auto& property : rwProperties) {
            configs.emplace_back(property);
        }
        return configs;
    }
//        else if (METRIC_KEY(IMPORT_EXPORT_SUPPORT) == name) {
//            return true;
//        }
    else if (ov::internal::caching_properties == name) {
        std::vector<ov::PropertyName> caching_properties = {ov::device::uuid};
        return decltype(ov::internal::caching_properties)::value_type(caching_properties);
    } else if (name == "SUPPORTED_METRICS") {  // TODO: Remove this key
        std::vector<std::string> configs;
        for (const auto& property : roProperties) {
            configs.emplace_back(property);
        }
        return configs;
    }
    OPENVINO_THROW("Unsupported property: ", name);
}

