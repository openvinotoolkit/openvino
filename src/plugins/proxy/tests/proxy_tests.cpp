// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "proxy_tests.hpp"

#include <memory>
#include <string>

#include "common_test_utils/file_utils.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/proxy/properties.hpp"
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
};

ov::PropertyName RW_property(const std::string& propertyName) {
    return ov::PropertyName(propertyName, ov::PropertyMutability::RW);
};

}  // namespace

void ov::proxy::tests::ProxyTests::SetUp() {
    if (m_mock_plugins.empty()) {
        register_plugin_support_reshape(core,
                                        "ABC",
                                        {{ov::proxy::configuration::alias.name(), "MOCK"},
                                         {ov::proxy::configuration::fallback.name(), "BDE"},
                                         {ov::proxy::configuration::priority.name(), 0}});
        register_plugin_support_subtract(core, "BDE", {{ov::proxy::configuration::alias.name(), "MOCK"}});
    }
}

void ov::proxy::tests::ProxyTests::TearDown() {
    ov::test::utils::removeDir("test_cache");
}

ov::Tensor ov::proxy::tests::ProxyTests::create_and_fill_tensor(const ov::element::Type& type, const ov::Shape& shape) {
    switch (type) {
    case ov::element::Type_t::i64:
        return create_tensor<ov::element_type_traits<ov::element::Type_t::i64>::value_type>(type, shape);
    default:
        break;
    }
    OPENVINO_THROW("Cannot generate tensor. Unsupported element type.");
}

std::shared_ptr<ov::Model> ov::proxy::tests::ProxyTests::create_model_with_subtract() {
    auto param = std::make_shared<ov::opset11::Parameter>(ov::element::i64, ov::Shape{1, 3, 2, 2});
    param->set_friendly_name("input");
    auto const_value = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
    const_value->set_friendly_name("const_val");
    auto add = std::make_shared<ov::opset11::Add>(param, const_value);
    add->set_friendly_name("add");
    auto subtract = std::make_shared<ov::opset11::Subtract>(add, const_value);
    subtract->set_friendly_name("sub");
    auto result = std::make_shared<ov::opset11::Result>(subtract);
    result->set_friendly_name("res");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
}

std::shared_ptr<ov::Model> ov::proxy::tests::ProxyTests::create_model_with_subtract_reshape() {
    auto param = std::make_shared<ov::opset11::Parameter>(ov::element::i64, ov::Shape{1, 3, 2, 2});
    param->set_friendly_name("input");
    auto const_value = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
    const_value->set_friendly_name("const_val");
    auto add = std::make_shared<ov::opset11::Add>(param, const_value);
    add->set_friendly_name("add");
    auto subtract = std::make_shared<ov::opset11::Subtract>(add, const_value);
    subtract->set_friendly_name("sub");
    auto reshape_val = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    reshape_val->set_friendly_name("reshape_val");
    auto reshape = std::make_shared<ov::opset11::Reshape>(subtract, reshape_val, true);
    reshape->set_friendly_name("reshape");
    auto result = std::make_shared<ov::opset11::Result>(reshape);
    result->set_friendly_name("res");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
}

std::shared_ptr<ov::Model> ov::proxy::tests::ProxyTests::create_model_with_add() {
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

std::shared_ptr<ov::Model> ov::proxy::tests::ProxyTests::create_model_with_subtract_reshape_relu() {
    auto param = std::make_shared<ov::opset11::Parameter>(ov::element::i64, ov::Shape{1, 3, 2, 2});
    param->set_friendly_name("input");
    auto const_value = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
    const_value->set_friendly_name("const_val");
    auto add = std::make_shared<ov::opset11::Add>(param, const_value);
    add->set_friendly_name("add");
    auto subtract = std::make_shared<ov::opset11::Subtract>(add, const_value);
    subtract->set_friendly_name("sub");
    auto reshape_val = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    reshape_val->set_friendly_name("reshape_val");
    auto reshape = std::make_shared<ov::opset11::Reshape>(subtract, reshape_val, true);
    reshape->set_friendly_name("reshape");
    auto relu = std::make_shared<ov::opset11::Relu>(reshape);
    relu->set_friendly_name("relu");
    auto result = std::make_shared<ov::opset11::Result>(relu);
    result->set_friendly_name("res");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
}

std::shared_ptr<ov::Model> ov::proxy::tests::ProxyTests::create_model_with_reshape() {
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
          m_has_context(false) {}

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
        for (const auto& input : get_inputs()) {
            input_tensors.emplace_back(ov::make_tensor(get_tensor(input)));
        }
        ov::TensorVector output_tensors;
        for (const auto& output : get_outputs()) {
            output_tensors.emplace_back(ov::make_tensor(get_tensor(output)));
        }
        m_model->evaluate(output_tensors, input_tensors);
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
};

std::shared_ptr<ov::ISyncInferRequest> MockCompiledModel::create_sync_infer_request() const {
    return std::make_shared<MockInferRequest>(std::dynamic_pointer_cast<const MockCompiledModel>(shared_from_this()));
}

class MockRemoteTensor : public ov::IRemoteTensor {
    ov::AnyMap m_properties;
    std::string m_dev_name;

public:
    MockRemoteTensor(const std::string& name, const ov::AnyMap& props) : m_properties(props), m_dev_name(name) {}
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
        OPENVINO_NOT_IMPLEMENTED;
    }

    const ov::Shape& get_shape() const override {
        OPENVINO_NOT_IMPLEMENTED;
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
        auto remote_tensor = std::make_shared<MockRemoteTensor>(m_dev_name, m_property);
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
        auto remote_tensor = std::make_shared<MockRemoteTensor>(m_dev_name, m_property);
        return {remote_tensor, nullptr};
    }
};

class MockPluginBase : public ov::IPlugin {
public:
    virtual const ov::Version& get_const_version() = 0;

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
        if (remote_properties.find("CUSTOM_CTX") == remote_properties.end())
            return std::make_shared<MockRemoteContext>(get_device_name());
        return std::make_shared<MockCustomRemoteContext>(get_device_name());
    }

    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remote_properties) const override {
        return std::make_shared<MockRemoteContext>(get_device_name());
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

void ov::proxy::tests::ProxyTests::reg_plugin(ov::Core& core,
                                              std::shared_ptr<ov::IPlugin>& plugin,
                                              const std::string& device_name,
                                              const ov::AnyMap& properties) {
    std::string libraryPath = get_mock_engine_path();
    if (!m_so)
        m_so = ov::util::load_shared_object(libraryPath.c_str());
    if (auto mock_plugin = std::dynamic_pointer_cast<MockPluginBase>(plugin))
        mock_plugin->set_version(mock_plugin->get_const_version());
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
void ov::proxy::tests::ProxyTests::register_plugin_support_reshape(ov::Core& core,
                                                                   const std::string& device_name,
                                                                   const ov::AnyMap& properties) {
    class MockPluginReshape : public MockPluginBase {
    public:
        const ov::Version& get_const_version() override {
            static const ov::Version version = {CI_BUILD_NUMBER, "openvino_mock_reshape_plugin"};
            return version;
        }
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
                    continue;
                else if (it.first == ov::cache_dir.name())
                    continue;
                else
                    OPENVINO_THROW(get_device_name(), " set config: " + it.first);
            }
        }

        ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override {
            const static std::vector<std::string> device_ids = {get_device_name() + "_1",
                                                                get_device_name() + "_2",
                                                                get_device_name() + "_3"};
            const static std::vector<ov::PropertyName> roProperties{
                RO_property(ov::supported_properties.name()),
                RO_property(ov::available_devices.name()),
                RO_property(ov::loaded_from_cache.name()),
                RO_property(ov::device::uuid.name()),
                RO_property(ov::device::capabilities.name()),
                RO_property(ov::optimal_batch_size.name()),
                RW_property(ov::hint::performance_mode.name()),
                RW_property(ov::hint::num_requests.name()),
            };
            // the whole config is RW before network is loaded.
            const static std::vector<ov::PropertyName> rwProperties{
                RW_property(ov::num_streams.name()),
                RW_property(ov::cache_dir.name()),
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
            } else if (name == ov::optimal_batch_size) {
                return decltype(ov::optimal_batch_size)::value_type{1};
            } else if (name == ov::hint::num_requests) {
                return decltype(ov::hint::num_requests)::value_type{1};
            } else if (name == ov::hint::performance_mode) {
                return decltype(ov::hint::performance_mode)::value_type{ov::hint::PerformanceMode::LATENCY};
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
            } else if (ov::internal::caching_properties == name) {
                std::vector<ov::PropertyName> caching_properties = {ov::device::uuid};
                return decltype(ov::internal::caching_properties)::value_type(caching_properties);
            } else if (name == ov::loaded_from_cache.name()) {
                return m_loaded_from_cache;
            } else if (name == ov::enable_profiling.name()) {
                return decltype(ov::enable_profiling)::value_type{m_profiling};
            } else if (name == ov::streams::num.name()) {
                return decltype(ov::streams::num)::value_type{num_streams};
            }
            OPENVINO_THROW("Unsupported property: ", name);
        }

    private:
        int32_t num_streams{0};
        bool m_profiling = false;
        bool m_loaded_from_cache{false};
    };

    auto plugin = std::make_shared<MockPluginReshape>();

    std::shared_ptr<ov::IPlugin> base_plugin = plugin;

    reg_plugin(core, base_plugin, device_name, properties);
}

void ov::proxy::tests::ProxyTests::register_plugin_support_subtract(ov::Core& core,
                                                                    const std::string& device_name,
                                                                    const ov::AnyMap& properties) {
    class MockPluginSubtract : public MockPluginBase {
    public:
        const ov::Version& get_const_version() override {
            static const ov::Version version = {CI_BUILD_NUMBER, "openvino_mock_subtract_plugin"};
            return version;
        }
        ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                        const ov::AnyMap& properties) const override {
            OPENVINO_ASSERT(model);

            std::unordered_set<std::string> supported_ops = {"Parameter", "Result", "Add", "Constant", "Subtract"};

            ov::SupportedOpsMap res;
            for (const auto& op : model->get_ordered_ops()) {
                if (supported_ops.find(op->get_type_info().name) == supported_ops.end())
                    continue;
                res[op->get_friendly_name()] = get_device_name();
            }
            return res;
        }

        void set_property(const ov::AnyMap& properties) override {
            for (const auto& it : properties) {
                if (it.first == ov::enable_profiling.name())
                    m_profiling = it.second.as<bool>();
                else if (it.first == ov::device::id.name() || it.first == ov::cache_dir.name())
                    continue;
                else
                    OPENVINO_THROW(get_device_name(), " set config: " + it.first);
            }
        }

        ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override {
            const static std::vector<std::string> device_ids = {get_device_name() + "_1",
                                                                get_device_name() + "_2",
                                                                get_device_name() + "_3"};
            const static std::vector<ov::PropertyName> roProperties{
                RO_property(ov::supported_properties.name()),
                RO_property(ov::available_devices.name()),
                RO_property(ov::loaded_from_cache.name()),
                RO_property(ov::device::uuid.name()),
                RO_property(ov::device::capabilities.name()),
            };
            // the whole config is RW before network is loaded.
            const static std::vector<ov::PropertyName> rwProperties{
                RW_property(ov::enable_profiling.name()),
                RW_property(ov::cache_dir.name()),
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
                    else if (device_id == device_ids[2])
                        uuid.uuid[i] = static_cast<uint8_t>(i * 5);
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
            } else if (ov::internal::caching_properties == name) {
                std::vector<ov::PropertyName> caching_properties = {ov::device::uuid};
                return decltype(ov::internal::caching_properties)::value_type(caching_properties);
            }
            OPENVINO_THROW("Unsupported property: ", name);
        }

    private:
        bool m_profiling{false};
        bool m_loaded_from_cache{false};
    };
    auto plugin = std::make_shared<MockPluginSubtract>();

    std::shared_ptr<ov::IPlugin> base_plugin = plugin;

    reg_plugin(core, base_plugin, device_name, properties);
}

void ov::proxy::tests::ProxyTests::register_plugin_without_devices(ov::Core& core,
                                                                   const std::string& device_name,
                                                                   const ov::AnyMap& properties) {
    class MockPluginNoDevices : public MockPluginBase {
    public:
        const ov::Version& get_const_version() override {
            static const ov::Version version = {CI_BUILD_NUMBER, "openvino_mock_no_devices_plugin"};
            return version;
        }
        ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                        const ov::AnyMap& properties) const override {
            OPENVINO_ASSERT(model);

            OPENVINO_NOT_IMPLEMENTED;
        }

        void set_property(const ov::AnyMap& properties) override {
            for (const auto& it : properties) {
                if (it.first == ov::enable_profiling.name())
                    m_profiling = it.second.as<bool>();
                else if (it.first == ov::device::id.name())
                    continue;
                else
                    OPENVINO_THROW(get_device_name(), " set config: " + it.first);
            }
        }

        ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override {
            const static std::vector<std::string> device_ids = {};
            const static std::vector<ov::PropertyName> roProperties{
                RO_property(ov::supported_properties.name()),
                RO_property(ov::available_devices.name()),
                RO_property(ov::loaded_from_cache.name()),
                RO_property(ov::device::capabilities.name()),
            };
            // the whole config is RW before network is loaded.
            const static std::vector<ov::PropertyName> rwProperties{
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
            }
            OPENVINO_THROW("Unsupported property: ", name);
        }

    private:
        bool m_profiling{false};
        bool m_loaded_from_cache{false};
    };
    auto plugin = std::make_shared<MockPluginNoDevices>();

    std::shared_ptr<ov::IPlugin> base_plugin = plugin;

    reg_plugin(core, base_plugin, device_name, properties);
}
