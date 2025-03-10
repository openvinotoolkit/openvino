// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "auto_func_test.hpp"

#include <chrono>
#include <memory>
#include <string>

#include "common_test_utils/file_utils.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/auto/properties.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/iremote_tensor.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

namespace {

std::string get_mock_engine_path() {
    std::string mockEngineName("mock_engine");
    return ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                              mockEngineName + OV_BUILD_POSTFIX);
}

template <class T>
std::function<T> make_std_function(const std::shared_ptr<void> so, const std::string& functionName) {
    std::function<T> ptr(reinterpret_cast<T*>(ov::util::get_symbol(so, functionName.c_str())));
    return ptr;
}

bool support_model(const std::shared_ptr<const ov::Model>& model, const ov::SupportedOpsMap& supported_ops) {
    for (const auto& op : model->get_ops()) {
        if (supported_ops.find(op->get_friendly_name()) == supported_ops.end())
            return false;
    }
    return true;
}

ov::PropertyName RO_property(const std::string& propertyName) {
    return ov::PropertyName(propertyName, ov::PropertyMutability::RO);
}

ov::PropertyName RW_property(const std::string& propertyName) {
    return ov::PropertyName(propertyName, ov::PropertyMutability::RW);
}

}  // namespace

void ov::auto_plugin::tests::AutoFuncTests::SetUp() {
    if (m_mock_plugins.empty()) {
        register_plugin_mock_cpu(core, "MOCK_CPU", {});
        register_plugin_mock_gpu(core, "MOCK_GPU", {});
        core.get_property("MOCK_CPU", ov::device::capabilities.name(), {});
        core.get_property("MOCK_GPU", ov::device::capabilities.name(), {});
    }
    model_can_batch = create_model_with_batch_possible();
    model_cannot_batch = create_model_with_reshape();
    auto hash = std::hash<std::string>()(::testing::UnitTest::GetInstance()->current_test_info()->name());
    std::stringstream ss;
    ss << std::this_thread::get_id();
    cache_path =
        "threading_test" + std::to_string(hash) + "_" + ss.str() + "_" + ov::test::utils::GetTimestamp() + "_cache";
}

void ov::auto_plugin::tests::AutoFuncTests::TearDown() {
    ov::test::utils::removeFilesWithExt(cache_path, "blob");
    ov::test::utils::removeDir(cache_path);
}

ov::Tensor ov::auto_plugin::tests::AutoFuncTests::create_and_fill_tensor(const ov::element::Type& type,
                                                                         const ov::Shape& shape) {
    switch (type) {
    case ov::element::Type_t::i64:
        return create_tensor<ov::element_type_traits<ov::element::Type_t::i64>::value_type>(type, shape);
    default:
        break;
    }
    OPENVINO_THROW("Cannot generate tensor. Unsupported element type.");
}

std::shared_ptr<ov::Model> ov::auto_plugin::tests::AutoFuncTests::create_model_with_batch_possible() {
    auto param = std::make_shared<ov::opset11::Parameter>(ov::element::i64, ov::Shape{1, 3, 2, 2});
    param->set_friendly_name("input");
    auto const_value = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
    const_value->set_friendly_name("const_val");
    auto add = std::make_shared<ov::opset11::Add>(param, const_value);
    add->set_friendly_name("add");
    auto result = std::make_shared<ov::opset11::Result>(add);
    result->set_friendly_name("res");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
}

std::shared_ptr<ov::Model> ov::auto_plugin::tests::AutoFuncTests::create_model_with_reshape() {
    auto param = std::make_shared<ov::opset11::Parameter>(ov::element::i64, ov::Shape{1, 3, 2, 2});
    param->set_friendly_name("input");
    auto const_value = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
    const_value->set_friendly_name("const_val");
    auto add = std::make_shared<ov::opset11::Add>(param, const_value);
    add->set_friendly_name("add");
    auto reshape_val = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    reshape_val->set_friendly_name("reshape_val");
    auto reshape = std::make_shared<ov::opset11::Reshape>(add, reshape_val, true);
    reshape->set_friendly_name("reshape");
    auto result = std::make_shared<ov::opset11::Result>(reshape);
    result->set_friendly_name("res");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
}

// Mock plugins

class MockCompiledModel : public ov::ICompiledModel {
public:
    MockCompiledModel(const std::shared_ptr<const ov::Model>& model,
                      const std::shared_ptr<const ov::IPlugin>& plugin,
                      const ov::AnyMap& config)
        : ov::ICompiledModel(model, plugin),
          m_config(config),
          m_model(model),
          m_has_context(false) {
        try {
            m_context = plugin->get_default_context(config);
        } catch (ov::Exception&) {
        }
    }

    MockCompiledModel(const std::shared_ptr<const ov::Model>& model,
                      const std::shared_ptr<const ov::IPlugin>& plugin,
                      const ov::AnyMap& config,
                      const ov::SoPtr<ov::IRemoteContext>& context)
        : ov::ICompiledModel(model, plugin),
          m_config(config),
          m_model(model),
          m_has_context(true),
          m_context(context) {}

    // Methods from a base class ov::ICompiledModel
    void export_model(std::ostream& model) const override {
        ov::pass::StreamSerialize(model, std::function<void(std::ostream&)>())
            .run_on_model(std::const_pointer_cast<ov::Model>(m_model));
    }

    std::shared_ptr<const ov::Model> get_runtime_model() const override {
        return m_model;
    }

    void set_property(const ov::AnyMap& properties) override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    ov::Any get_property(const std::string& name) const override {
        auto prop = m_config.find(name);
        if (prop != m_config.end())
            return prop->second;
        if (name == ov::supported_properties) {
            std::vector<ov::PropertyName> supportedProperties{ov::optimal_number_of_infer_requests,
                                                              ov::hint::performance_mode};

            return decltype(ov::supported_properties)::value_type(supportedProperties);
        } else if (name == ov::optimal_number_of_infer_requests.name()) {
            return decltype(ov::optimal_number_of_infer_requests)::value_type(2);
        } else if (name == ov::model_name) {
            return decltype(ov::model_name)::value_type(m_model->get_name());
        } else if (name == ov::execution_devices) {
            return decltype(ov::execution_devices)::value_type({get_plugin()->get_device_name()});
        }
        OPENVINO_NOT_IMPLEMENTED;
    }

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

    const std::shared_ptr<const ov::Model>& get_model() const {
        return m_model;
    }

    ov::SoPtr<ov::IRemoteContext> get_context() const {
        return m_context;
    }

    bool has_context() const {
        return m_has_context;
    }

private:
    ov::AnyMap m_config;
    std::shared_ptr<const ov::Model> m_model;
    bool m_has_context;
    ov::SoPtr<ov::IRemoteContext> m_context;
};

class MockInferRequest : public ov::ISyncInferRequest {
public:
    MockInferRequest(const std::shared_ptr<const MockCompiledModel>& compiled_model)
        : ov::ISyncInferRequest(compiled_model) {
        OPENVINO_ASSERT(compiled_model);
        m_model = compiled_model->get_model();
        m_has_context = compiled_model->get_context() != nullptr;
        // Allocate input/output tensors
        for (const auto& input : get_inputs()) {
            allocate_tensor(input, [this, input, compiled_model](ov::SoPtr<ov::ITensor>& tensor) {
                // Can add a check to avoid double work in case of shared tensors
                allocate_tensor_impl(tensor,
                                     input.get_element_type(),
                                     input.get_partial_shape().is_dynamic() ? ov::Shape{0} : input.get_shape(),
                                     compiled_model->has_context(),
                                     compiled_model->get_context());
            });
        }
        for (const auto& output : get_outputs()) {
            allocate_tensor(output, [this, output, compiled_model](ov::SoPtr<ov::ITensor>& tensor) {
                // Can add a check to avoid double work in case of shared tensors
                allocate_tensor_impl(tensor,
                                     output.get_element_type(),
                                     output.get_partial_shape().is_dynamic() ? ov::Shape{0} : output.get_shape(),
                                     compiled_model->has_context(),
                                     compiled_model->get_context());
            });
        }
    }
    ~MockInferRequest() = default;

    void infer() override {
        ov::TensorVector input_tensors;
        bool evaludate_flag = true;
        for (const auto& input : get_inputs()) {
            auto tensor = get_tensor(input);
            // check if valid if remote tensor
            if (std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor._ptr) && m_has_context) {
                evaludate_flag = false;
                auto remote_tensor = std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor._ptr);
                if (remote_tensor->get_device_name() != get_compiled_model()->get_context()->get_device_name())
                    OPENVINO_THROW("cannot consume the buffer!");
            }
            input_tensors.emplace_back(ov::make_tensor(tensor));
        }
        ov::TensorVector output_tensors;
        for (const auto& output : get_outputs()) {
            auto tensor = get_tensor(output);
            // check if valid if remote tensor
            if (std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor._ptr) && m_has_context) {
                evaludate_flag = false;
                auto remote_tensor = std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor._ptr);
                if (remote_tensor->get_device_name() != get_compiled_model()->get_context()->get_device_name())
                    OPENVINO_THROW("cannot consume the buffer!");
            }
            output_tensors.emplace_back(ov::make_tensor(tensor));
        }
        if (evaludate_flag) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));  // add delay for test
            m_model->evaluate(output_tensors, input_tensors);
        }
    }
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override {
        OPENVINO_NOT_IMPLEMENTED;
    }
    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

private:
    void allocate_tensor_impl(ov::SoPtr<ov::ITensor>& tensor,
                              const ov::element::Type& element_type,
                              const ov::Shape& shape,
                              bool has_context,
                              ov::SoPtr<ov::IRemoteContext> context) {
        if (!tensor || tensor->get_element_type() != element_type) {
            if (has_context) {
                tensor = context->create_tensor(element_type, shape, {});
            } else {
                tensor = ov::SoPtr<ov::ITensor>(ov::make_tensor(element_type, shape), nullptr);
            }
        } else {
            tensor->set_shape(shape);
        }
    }
    std::shared_ptr<const ov::Model> m_model;
    bool m_has_context;
};

std::shared_ptr<ov::ISyncInferRequest> MockCompiledModel::create_sync_infer_request() const {
    return std::make_shared<MockInferRequest>(std::dynamic_pointer_cast<const MockCompiledModel>(shared_from_this()));
}

class MockRemoteTensor : public ov::IRemoteTensor {
    ov::AnyMap m_properties;
    std::string m_dev_name;
    ov::element::Type m_element_type;
    ov::Shape m_shape;

public:
    MockRemoteTensor(const std::string& name,
                     const ov::AnyMap& props,
                     const ov::element::Type& type,
                     const ov::Shape& shape)
        : m_properties(props),
          m_dev_name(name),
          m_element_type(type),
          m_shape(shape) {}

    const ov::AnyMap& get_properties() const override {
        return m_properties;
    }
    const std::string& get_device_name() const override {
        return m_dev_name;
    }
    void set_shape(ov::Shape shape) override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    const ov::element::Type& get_element_type() const override {
        return m_element_type;
    }

    const ov::Shape& get_shape() const override {
        return m_shape;
    }

    const ov::Strides& get_strides() const override {
        OPENVINO_NOT_IMPLEMENTED;
    }
};

class MockRemoteContext : public ov::IRemoteContext {
    ov::AnyMap m_property = {{"IS_DEFAULT", true}};
    std::string m_dev_name;

public:
    MockRemoteContext(const std::string& dev_name) : m_dev_name(dev_name) {}
    const std::string& get_device_name() const override {
        return m_dev_name;
    }

    const ov::AnyMap& get_property() const override {
        return m_property;
    }

    ov::SoPtr<ov::IRemoteTensor> create_tensor(const ov::element::Type& type,
                                               const ov::Shape& shape,
                                               const ov::AnyMap& params = {}) override {
        auto remote_tensor = std::make_shared<MockRemoteTensor>(m_dev_name, m_property, type, shape);
        return {remote_tensor, nullptr};
    }
};

class MockCustomRemoteContext : public ov::IRemoteContext {
    ov::AnyMap m_property = {{"IS_DEFAULT", false}};
    std::string m_dev_name;

public:
    MockCustomRemoteContext(const std::string& dev_name) : m_dev_name(dev_name) {}
    const std::string& get_device_name() const override {
        return m_dev_name;
    }

    const ov::AnyMap& get_property() const override {
        return m_property;
    }

    ov::SoPtr<ov::IRemoteTensor> create_tensor(const ov::element::Type& type,
                                               const ov::Shape& shape,
                                               const ov::AnyMap& params = {}) override {
        auto remote_tensor = std::make_shared<MockRemoteTensor>(m_dev_name, m_property, type, shape);
        return {remote_tensor, nullptr};
    }
};

class MockPluginBase : public ov::IPlugin {
public:
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties) const override {
        OPENVINO_ASSERT(model);
        if (!support_model(model, query_model(model, properties)))
            OPENVINO_THROW("Unsupported model");

        return std::make_shared<MockCompiledModel>(model, shared_from_this(), properties);
    }

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::string& model_path,
                                                      const ov::AnyMap& properties) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties,
                                                      const ov::SoPtr<ov::IRemoteContext>& context) const override {
        if (!support_model(model, query_model(model, properties)))
            OPENVINO_THROW("Unsupported model");

        return std::make_shared<MockCompiledModel>(model, shared_from_this(), properties, context);
    }

    void set_property(const ov::AnyMap& properties) override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap& remote_properties) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remote_properties) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model, const ov::AnyMap& properties) const override {
        std::string xmlString, xmlInOutString;
        ov::Tensor weights;

        ov::pass::StreamSerialize::DataHeader hdr = {};
        model.read(reinterpret_cast<char*>(&hdr), sizeof hdr);

        // read CNNNetwork input/output precisions
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

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                     const ov::AnyMap& properties) const override {
        std::string xmlString, xmlInOutString;
        ov::Tensor weights;

        ov::pass::StreamSerialize::DataHeader hdr = {};
        model.read(reinterpret_cast<char*>(&hdr), sizeof hdr);

        // read CNNNetwork input/output precisions
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
        return compile_model(ov_model, properties, context);
    }

    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }
};

class MockPluginSupportBatchAndContext : public MockPluginBase {
public:
    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const override {
        OPENVINO_ASSERT(model);

        std::unordered_set<std::string> supported_ops = {"Parameter", "Result", "Add", "Constant", "Reshape"};

        ov::SupportedOpsMap res;
        for (const auto& op : model->get_ordered_ops()) {
            if (supported_ops.find(op->get_type_info().name) == supported_ops.end())
                continue;
            res.emplace(op->get_friendly_name(), get_device_name());
        }
        return res;
    }

    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap& remote_properties) const override {
        if (remote_properties.find("CUSTOM_CTX") == remote_properties.end())
            return std::make_shared<MockRemoteContext>(get_device_name());
        return std::make_shared<MockCustomRemoteContext>(get_device_name());
    }

    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remote_properties) const override {
        std::string device_name = get_device_name();
        if (remote_properties.find(ov::device::id.name()) != remote_properties.end())
            device_name = device_name + "." + remote_properties.at(ov::device::id.name()).as<std::string>();

        return std::make_shared<MockRemoteContext>(device_name);
    }

    void set_property(const ov::AnyMap& properties) override {
        for (const auto& it : properties) {
            if (it.first == ov::num_streams.name())
                num_streams = it.second.as<int32_t>();
            else if (it.first == ov::enable_profiling.name())
                m_profiling = it.second.as<bool>();
            else if (it.first == ov::hint::performance_mode.name())
                m_perf_hint = it.second.as<ov::hint::PerformanceMode>();
            else if (it.first == ov::hint::num_requests.name())
                m_request = it.second.as<uint32_t>();
            else if (it.first == ov::device::id.name())
                m_id = it.second.as<std::string>();
            else if (it.first == ov::cache_dir.name())
                continue;
            else
                OPENVINO_THROW(get_device_name(), " set config: " + it.first);
        }
    }

    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override {
        const std::vector<ov::PropertyName> roProperties{RO_property(ov::supported_properties.name()),
                                                         RO_property(ov::optimal_batch_size.name()),
                                                         RO_property(ov::device::capabilities.name()),
                                                         RO_property(ov::device::type.name()),
                                                         RO_property(ov::device::uuid.name()),
                                                         RO_property(ov::device::id.name()),
                                                         RO_property(ov::available_devices.name()),
                                                         RO_property(ov::intel_gpu::memory_statistics.name())};
        // the whole config is RW before network is loaded.
        const std::vector<ov::PropertyName> rwProperties{RW_property(ov::num_streams.name()),
                                                         RW_property(ov::enable_profiling.name()),
                                                         RW_property(ov::compilation_num_threads.name()),
                                                         RW_property(ov::hint::performance_mode.name()),
                                                         RW_property(ov::hint::num_requests.name())};
        if (name == ov::supported_properties) {
            std::vector<ov::PropertyName> supportedProperties;
            supportedProperties.reserve(roProperties.size() + rwProperties.size());
            supportedProperties.insert(supportedProperties.end(), roProperties.begin(), roProperties.end());
            supportedProperties.insert(supportedProperties.end(), rwProperties.begin(), rwProperties.end());

            return decltype(ov::supported_properties)::value_type(supportedProperties);
        } else if (name == ov::hint::num_requests.name()) {
            return decltype(ov::hint::num_requests)::value_type(1);
        } else if (name == ov::hint::performance_mode.name()) {
            return decltype(ov::hint::performance_mode)::value_type(ov::hint::PerformanceMode::LATENCY);
        } else if (name == ov::optimal_batch_size.name()) {
            return decltype(ov::optimal_batch_size)::value_type(4);
        } else if (name == ov::device::capabilities.name()) {
            return decltype(ov::device::capabilities)::value_type(
                {"FP32", "FP16", "BIN", "INT8", ov::device::capability::EXPORT_IMPORT});
        } else if (name == ov::device::type.name()) {
            return decltype(ov::device::type)::value_type(ov::device::Type::INTEGRATED);
        } else if (name == ov::loaded_from_cache.name()) {
            return false;
        } else if (name == ov::enable_profiling.name()) {
            return decltype(ov::enable_profiling)::value_type{false};
        } else if (name == ov::streams::num.name()) {
            return decltype(ov::streams::num)::value_type{2};
        } else if (name == ov::compilation_num_threads.name()) {
            return decltype(ov::compilation_num_threads)::value_type{4};
        } else if (name == ov::internal::supported_properties) {
            return decltype(ov::internal::supported_properties)::value_type(
                {ov::PropertyName{ov::internal::caching_properties.name(), ov::PropertyMutability::RO}});
        } else if (ov::internal::caching_properties == name) {
            std::vector<ov::PropertyName> caching_properties = {ov::device::uuid, ov::device::id};
            return decltype(ov::internal::caching_properties)::value_type(caching_properties);
        } else if (name == ov::device::uuid) {
            ov::device::UUID uuid = {};
            return decltype(ov::device::uuid)::value_type{uuid};
        } else if (name == ov::device::id) {
            return decltype(ov::device::id)::value_type{m_id};
        } else if (name == ov::available_devices.name()) {
            std::vector<std::string> available_devices = {};
            return decltype(ov::available_devices)::value_type(available_devices);
        } else if (name == ov::loaded_from_cache.name()) {
            return m_loaded_from_cache;
        } else if (name == ov::intel_gpu::memory_statistics) {
            return decltype(ov::intel_gpu::memory_statistics)::value_type{{}};
        }
        OPENVINO_NOT_IMPLEMENTED;
    }

private:
    int32_t num_streams{0};
    bool m_profiling = false;
    bool m_loaded_from_cache{false};
    ov::hint::PerformanceMode m_perf_hint = ov::hint::PerformanceMode::THROUGHPUT;
    uint32_t m_request = 0;
    std::string m_id;
};

void ov::auto_plugin::tests::AutoFuncTests::reg_plugin(ov::Core& core,
                                                       std::shared_ptr<ov::IPlugin>& plugin,
                                                       const std::string& device_name,
                                                       const ov::AnyMap& properties) {
    std::string libraryPath = get_mock_engine_path();
    if (!m_so)
        m_so = ov::util::load_shared_object(libraryPath.c_str());
    plugin->set_device_name(device_name);
    std::function<void(ov::IPlugin*)> injectProxyEngine = make_std_function<void(ov::IPlugin*)>(m_so, "InjectPlugin");

    injectProxyEngine(plugin.get());
    core.register_plugin(ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                                            std::string("mock_engine") + OV_BUILD_POSTFIX),
                         device_name,
                         properties);
    m_mock_plugins.emplace_back(plugin);
}

// test
void ov::auto_plugin::tests::AutoFuncTests::register_plugin_mock_gpu(ov::Core& core,
                                                                     const std::string& device_name,
                                                                     const ov::AnyMap& properties) {
    std::shared_ptr<ov::IPlugin> base_plugin = std::make_shared<MockPluginSupportBatchAndContext>();
    reg_plugin(core, base_plugin, device_name, properties);
}

class MockPlugin : public MockPluginBase {
public:
    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const override {
        OPENVINO_ASSERT(model);

        std::unordered_set<std::string> supported_ops = {"Parameter", "Result", "Add", "Constant", "Reshape"};

        ov::SupportedOpsMap res;
        for (const auto& op : model->get_ordered_ops()) {
            if (supported_ops.find(op->get_type_info().name) == supported_ops.end())
                continue;
            res.emplace(op->get_friendly_name(), get_device_name());
        }
        return res;
    }

    void set_property(const ov::AnyMap& properties) override {
        for (const auto& it : properties) {
            if (it.first == ov::num_streams.name())
                num_streams = it.second.as<int32_t>();
            else if (it.first == ov::enable_profiling.name())
                m_profiling = it.second.as<bool>();
            else if (it.first == ov::device::id.name())
                m_id = it.second.as<std::string>();
            else if (it.first == ov::cache_dir.name())
                continue;
            else
                OPENVINO_THROW(get_device_name(), " set config: " + it.first);
        }
    }

    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override {
        const std::vector<ov::PropertyName> roProperties{RO_property(ov::supported_properties.name()),
                                                         RO_property(ov::device::uuid.name()),
                                                         RO_property(ov::device::id.name()),
                                                         RO_property(ov::available_devices.name()),
                                                         RO_property(ov::device::capabilities.name())};
        // the whole config is RW before network is loaded.
        const std::vector<ov::PropertyName> rwProperties{RW_property(ov::num_streams.name()),
                                                         RW_property(ov::enable_profiling.name()),
                                                         RW_property(ov::hint::performance_mode.name())};
        if (name == ov::supported_properties) {
            std::vector<ov::PropertyName> supportedProperties;
            supportedProperties.reserve(roProperties.size() + rwProperties.size());
            supportedProperties.insert(supportedProperties.end(), roProperties.begin(), roProperties.end());
            supportedProperties.insert(supportedProperties.end(), rwProperties.begin(), rwProperties.end());

            return decltype(ov::supported_properties)::value_type(supportedProperties);
        } else if (name == ov::loaded_from_cache.name()) {
            return false;
        } else if (name == ov::enable_profiling.name()) {
            return decltype(ov::enable_profiling)::value_type{false};
        } else if (name == ov::streams::num.name()) {
            return decltype(ov::streams::num)::value_type{2};
        } else if (name == ov::internal::supported_properties) {
            return decltype(ov::internal::supported_properties)::value_type(
                {ov::PropertyName{ov::internal::caching_properties.name(), ov::PropertyMutability::RO}});
        } else if (name == ov::device::capabilities) {
            std::vector<std::string> capabilities;
            capabilities.push_back(ov::device::capability::EXPORT_IMPORT);
            return decltype(ov::device::capabilities)::value_type(capabilities);
        } else if (ov::internal::caching_properties == name) {
            std::vector<ov::PropertyName> caching_properties = {ov::device::uuid, ov::device::id};
            return decltype(ov::internal::caching_properties)::value_type(caching_properties);
        } else if (name == ov::device::uuid) {
            ov::device::UUID uuid = {};
            return decltype(ov::device::uuid)::value_type{uuid};
        } else if (name == ov::device::id) {
            return decltype(ov::device::id)::value_type{m_id};
        } else if (name == ov::available_devices.name()) {
            std::vector<std::string> available_devices = {};
            return decltype(ov::available_devices)::value_type(available_devices);
        } else if (name == ov::loaded_from_cache.name()) {
            return m_loaded_from_cache;
        }
        OPENVINO_NOT_IMPLEMENTED;
    }

private:
    int32_t num_streams{0};
    bool m_profiling = false;
    bool m_loaded_from_cache{false};
    std::string m_id;
};

void ov::auto_plugin::tests::AutoFuncTests::register_plugin_mock_cpu(ov::Core& core,
                                                                     const std::string& device_name,
                                                                     const ov::AnyMap& properties) {
    std::shared_ptr<ov::IPlugin> base_plugin = std::make_shared<MockPlugin>();

    reg_plugin(core, base_plugin, device_name, properties);
}

void ov::auto_plugin::tests::AutoFuncTests::register_plugin_mock_gpu_compile_slower(ov::Core& core,
                                                                                    const std::string& device_name,
                                                                                    const ov::AnyMap& properties) {
    class MockPluginCompileSlower : public MockPluginSupportBatchAndContext {
    public:
        std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                          const ov::AnyMap& properties) const override {
            OPENVINO_ASSERT(model);
            if (!support_model(model, query_model(model, properties)))
                OPENVINO_THROW("Unsupported model");
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));  // add delay for test
            return std::make_shared<MockCompiledModel>(model, shared_from_this(), properties);
        }
        std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                          const ov::AnyMap& properties,
                                                          const ov::SoPtr<ov::IRemoteContext>& context) const override {
            if (!support_model(model, query_model(model, properties)))
                OPENVINO_THROW("Unsupported model");
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));  // add delay for test
            return std::make_shared<MockCompiledModel>(model, shared_from_this(), properties, context);
        }
    };

    std::shared_ptr<ov::IPlugin> base_plugin = std::make_shared<MockPluginCompileSlower>();
    reg_plugin(core, base_plugin, device_name, properties);
}

void ov::auto_plugin::tests::AutoFuncTests::register_plugin_mock_cpu_compile_slower(ov::Core& core,
                                                                                    const std::string& device_name,
                                                                                    const ov::AnyMap& properties) {
    class MockCPUPluginCompileSlower : public MockPlugin {
    public:
        std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                          const ov::AnyMap& properties) const override {
            OPENVINO_ASSERT(model);
            if (!support_model(model, query_model(model, properties)))
                OPENVINO_THROW("Unsupported model");
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));  // add delay for test
            return std::make_shared<MockCompiledModel>(model, shared_from_this(), properties);
        }
        std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                          const ov::AnyMap& properties,
                                                          const ov::SoPtr<ov::IRemoteContext>& context) const override {
            if (!support_model(model, query_model(model, properties)))
                OPENVINO_THROW("Unsupported model");
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));  // add delay for test
            return std::make_shared<MockCompiledModel>(model, shared_from_this(), properties, context);
        }
    };

    std::shared_ptr<ov::IPlugin> base_plugin = std::make_shared<MockCPUPluginCompileSlower>();
    reg_plugin(core, base_plugin, device_name, properties);
}