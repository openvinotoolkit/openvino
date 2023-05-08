// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "proxy_tests.hpp"

#include <gmock/gmock.h>

#include "common_test_utils/file_utils.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

void ov::proxy::tests::ProxyTests::SetUp() {
    if (m_mock_plugins.empty()) {
        register_plugin_support_reshape(core,
                                        "ABC",
                                        {{"ALIAS", "MOCK"}, {"FALLBACK", "BDE"}, {"DEVICE_PRIORITY", "0"}});
        register_plugin_support_subtract(core, "BDE", {{"ALIAS", "MOCK"}});
    }
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

using namespace ::testing;

class MockPluginBase : public ov::IPlugin {
public:
    MOCK_CONST_METHOD2(compile_model,
                       std::shared_ptr<ov::ICompiledModel>(const std::shared_ptr<const ov::Model>& model,
                                                           const ov::AnyMap& properties));
    MOCK_CONST_METHOD3(compile_model,
                       std::shared_ptr<ov::ICompiledModel>(const std::shared_ptr<const ov::Model>& model,
                                                           const ov::AnyMap& properties,
                                                           const ov::RemoteContext& context));
    MOCK_CONST_METHOD2(compile_model,
                       std::shared_ptr<ov::ICompiledModel>(const std::string& model_path,
                                                           const ov::AnyMap& properties));

    MOCK_METHOD1(set_property, void(const ov::AnyMap& properties));

    MOCK_CONST_METHOD2(get_property, ov::Any(const std::string& name, const ov::AnyMap& arguments));

    MOCK_CONST_METHOD1(create_context, std::shared_ptr<ov::IRemoteContext>(const ov::AnyMap& remote_properties));

    MOCK_CONST_METHOD1(get_default_context, std::shared_ptr<ov::IRemoteContext>(const ov::AnyMap& remote_properties));

    MOCK_CONST_METHOD2(import_model,
                       std::shared_ptr<ov::ICompiledModel>(std::istream& model, const ov::AnyMap& properties));
    MOCK_CONST_METHOD3(import_model,
                       std::shared_ptr<ov::ICompiledModel>(std::istream& model,
                                                           const ov::RemoteContext& context,
                                                           const ov::AnyMap& properties));

    MOCK_CONST_METHOD2(query_model,
                       ov::SupportedOpsMap(const std::shared_ptr<const ov::Model>& model,
                                           const ov::AnyMap& properties));
};

class MockInferRequest : public ov::ISyncInferRequest {
public:
    MockInferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled_model)
        : ov::ISyncInferRequest(compiled_model) {}
    ~MockInferRequest() = default;

    void infer() override {
        OPENVINO_NOT_IMPLEMENTED;
    }
    std::vector<std::shared_ptr<ov::IVariableState>> query_state() const override {
        OPENVINO_NOT_IMPLEMENTED;
    }
    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        OPENVINO_NOT_IMPLEMENTED;
    }
};

class MockCompiledModel : public ov::ICompiledModel {
public:
    MockCompiledModel(const std::shared_ptr<const ov::Model>& model,
                      const std::shared_ptr<const ov::IPlugin>& plugin,
                      const ov::AnyMap& config)
        : ov::ICompiledModel(model, plugin),
          m_config(config) {}

    // Methods from a base class ov::ICompiledModel
    void export_model(std::ostream& model) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    std::shared_ptr<const ov::Model> get_runtime_model() const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    void set_property(const ov::AnyMap& properties) override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    ov::Any get_property(const std::string& name) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override {
        return std::make_shared<MockInferRequest>(shared_from_this());
    }

private:
    friend MockInferRequest;
    ov::AnyMap m_config;
};

namespace {
std::string get_mock_engine_path() {
    std::string mockEngineName("mock_engine");
    return ov::util::make_plugin_library_name(CommonTestUtils::getExecutableDirectory(),
                                              mockEngineName + IE_BUILD_POSTFIX);
}

template <class T>
std::function<T> make_std_function(const std::shared_ptr<void> so, const std::string& functionName) {
    std::function<T> ptr(reinterpret_cast<T*>(ov::util::get_symbol(so, functionName.c_str())));
    return ptr;
}

std::shared_ptr<ov::IPlugin> reg_plugin(ov::Core& core,
                                        std::shared_ptr<ov::IPlugin>& plugin,
                                        const std::string& device_name,
                                        const ov::AnyMap& properties) {
    std::string libraryPath = get_mock_engine_path();
    std::shared_ptr<void> sharedObjectLoader = ov::util::load_shared_object(libraryPath.c_str());
    std::function<void(ov::IPlugin*)> injectProxyEngine =
        make_std_function<void(ov::IPlugin*)>(sharedObjectLoader, "InjectPlugin");

    injectProxyEngine(plugin.get());
    core.register_plugin(ov::util::make_plugin_library_name(CommonTestUtils::getExecutableDirectory(),
                                                            std::string("mock_engine") + IE_BUILD_POSTFIX),
                         device_name,
                         properties);
    return plugin;
}

bool support_model(const std::shared_ptr<const ov::Model>& model, const ov::SupportedOpsMap& supported_ops) {
    for (const auto& op : model->get_ops()) {
        if (supported_ops.find(op->get_friendly_name()) == supported_ops.end())
            return false;
    }
    return true;
}
}  // namespace

// test
void ov::proxy::tests::ProxyTests::register_plugin_support_reshape(ov::Core& core,
                                                                   const std::string& device_name,
                                                                   const ov::AnyMap& properties) {
    auto plugin = std::make_shared<MockPluginBase>();

    const auto& query_ov_model = [&](const std::shared_ptr<const ov::Model>& model,
                                     const ov::AnyMap& properties) -> ov::SupportedOpsMap {
        OPENVINO_ASSERT(model);

        std::unordered_set<std::string> supported_ops = {"Parameter", "Result", "Add", "Constant", "Reshape"};

        ov::SupportedOpsMap res;
        for (const auto& op : model->get_ordered_ops()) {
            if (supported_ops.find(op->get_type_info().name) == supported_ops.end())
                continue;
            res.emplace(op->get_friendly_name(), plugin->get_device_name());
        }
        return res;
    };

    ON_CALL(*plugin, query_model(_, _)).WillByDefault(Invoke(query_ov_model));

    ON_CALL(*plugin, compile_model(Matcher<const std::shared_ptr<const ov::Model>&>(_), _))
        .WillByDefault(Invoke([&](const std::shared_ptr<const ov::Model>& model,
                                  const ov::AnyMap& properties) -> std::shared_ptr<ov::ICompiledModel> {
            OPENVINO_ASSERT(model);
            if (!support_model(model, query_ov_model(model, properties)))
                OPENVINO_THROW("Unsupported model");

            return std::make_shared<MockCompiledModel>(model, plugin, properties);
        }));
    std::shared_ptr<ov::IPlugin> base_plugin = plugin;

    m_mock_plugins.emplace_back(reg_plugin(core, base_plugin, device_name, properties));
}

void ov::proxy::tests::ProxyTests::register_plugin_support_subtract(ov::Core& core,
                                                                    const std::string& device_name,
                                                                    const ov::AnyMap& properties) {
    auto plugin = std::make_shared<MockPluginBase>();

    const auto& query_ov_model = [&](const std::shared_ptr<const ov::Model>& model,
                                     const ov::AnyMap& properties) -> ov::SupportedOpsMap {
        OPENVINO_ASSERT(model);

        std::unordered_set<std::string> supported_ops = {"Parameter", "Result", "Add", "Constant", "Subtract"};

        ov::SupportedOpsMap res;
        for (const auto& op : model->get_ordered_ops()) {
            if (supported_ops.find(op->get_type_info().name) == supported_ops.end())
                continue;
            res.emplace(op->get_friendly_name(), plugin->get_device_name());
        }
        return res;
    };

    ON_CALL(*plugin, query_model(_, _)).WillByDefault(Invoke(query_ov_model));

    ON_CALL(*plugin, compile_model(Matcher<const std::shared_ptr<const ov::Model>&>(_), _))
        .WillByDefault(Invoke([&](const std::shared_ptr<const ov::Model>& model,
                                  const ov::AnyMap& properties) -> std::shared_ptr<ov::ICompiledModel> {
            OPENVINO_ASSERT(model);
            if (!support_model(model, query_ov_model(model, properties)))
                OPENVINO_THROW("Unsupported model");

            return std::make_shared<MockCompiledModel>(model, plugin, properties);
        }));
    std::shared_ptr<ov::IPlugin> base_plugin = plugin;

    m_mock_plugins.emplace_back(reg_plugin(core, base_plugin, device_name, properties));
}
