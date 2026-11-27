// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "serialization.hpp"

#define private public
#include "compiled_model.hpp"
#undef private

#include "unit_test_utils/mocks/openvino/runtime/mock_icore.hpp"

namespace {

class TestPlugin final : public ov::IPlugin {
public:
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>&,
                                                      const ov::AnyMap&) const override {
        OPENVINO_THROW("Unexpected TestPlugin::compile_model call");
    }
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>&,
                                                      const ov::AnyMap&,
                                                      const ov::SoPtr<ov::IRemoteContext>&) const override {
        OPENVINO_THROW("Unexpected TestPlugin::compile_model(context) call");
    }
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream&, const ov::AnyMap&) const override {
        OPENVINO_THROW("Unexpected TestPlugin::import_model(stream) call");
    }
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream&,
                                                     const ov::SoPtr<ov::IRemoteContext>&,
                                                     const ov::AnyMap&) const override {
        OPENVINO_THROW("Unexpected TestPlugin::import_model(stream, context) call");
    }
    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor&, const ov::AnyMap&) const override {
        OPENVINO_THROW("Unexpected TestPlugin::import_model(blob) call");
    }
    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor&,
                                                     const ov::SoPtr<ov::IRemoteContext>&,
                                                     const ov::AnyMap&) const override {
        OPENVINO_THROW("Unexpected TestPlugin::import_model(blob, context) call");
    }
    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>&, const ov::AnyMap&) const override {
        OPENVINO_THROW("Unexpected TestPlugin::query_model call");
    }
    void set_property(const ov::AnyMap&) override {}
    ov::Any get_property(const std::string&, const ov::AnyMap&) const override {
        OPENVINO_THROW("Test plugin does not expose properties");
    }
    bool is_property_supported(const std::string&, const ov::AnyMap&) const override {
        return false;
    }
    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap&) const override {
        OPENVINO_THROW("Unexpected TestPlugin::create_context call");
    }
    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap&) const override {
        OPENVINO_THROW("Unexpected TestPlugin::get_default_context call");
    }
};

class FakeSubCompiledModel final : public ov::ICompiledModel {
public:
    FakeSubCompiledModel(const std::shared_ptr<ov::Model>& model,
                         const std::shared_ptr<const ov::IPlugin>& plugin,
                         std::string export_blob = {},
                         std::string device = "NPU")
        : ov::ICompiledModel(model, plugin, nullptr, nullptr),
          m_model(model),
          m_export_blob(std::move(export_blob)),
          m_device(std::move(device)) {}

    void export_model(std::ostream& stream) const override {
        stream.write(m_export_blob.data(), static_cast<std::streamsize>(m_export_blob.size()));
    }
    std::shared_ptr<const ov::Model> get_runtime_model() const override {
        return m_model;
    }
    void set_property(const ov::AnyMap&) override {}
    ov::Any get_property(const std::string& name) const override {
        if (name == ov::execution_devices.name()) {
            return std::vector<std::string>{m_device};
        }
        return {};
    }

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override {
        return {};
    }
    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override {
        return {};
    }

private:
    std::shared_ptr<ov::Model> m_model;
    std::string m_export_blob;
    std::string m_device;
};

std::pair<std::shared_ptr<TestPlugin>, std::shared_ptr<testing::NiceMock<ov::MockICore>>> make_test_plugin_with_core() {
    auto plugin = std::make_shared<TestPlugin>();
    auto core = std::make_shared<testing::NiceMock<ov::MockICore>>();

    ON_CALL(*core, get_supported_property(testing::_, testing::_, testing::_))
        .WillByDefault([](const std::string&, const ov::AnyMap& properties, const bool) {
            return properties;
        });
    ON_CALL(*core, get_property(testing::_, testing::_, testing::_))
        .WillByDefault([](const std::string&, const std::string& name, const ov::AnyMap&) -> ov::Any {
            if (name == ov::available_devices.name()) {
                return std::vector<std::string>{};
            }
            if (name == ov::intel_npu::compiler_version.name()) {
                return int64_t{0};
            }
            if (name == ov::device::architecture.name()) {
                return std::string{};
            }
            if (name == ov::supported_properties.name() || name == ov::internal::supported_properties.name()) {
                return std::vector<ov::PropertyName>{};
            }
            return {};
        });
    ON_CALL(*core, get_property(testing::_, testing::_))
        .WillByDefault([](const std::string&, const std::string& name) -> ov::Any {
            if (name == ov::available_devices.name()) {
                return std::vector<std::string>{};
            }
            if (name == ov::supported_properties.name()) {
                return std::vector<ov::PropertyName>{};
            }
            if (name == ov::intel_npu::compiler_version.name()) {
                return static_cast<int64_t>(0);
            }
            if (name == ov::device::architecture.name()) {
                return std::string{};
            }
            return {};
        });

    plugin->set_core(core);
    return {plugin, core};
}

std::shared_ptr<TestPlugin> make_test_plugin() {
    return make_test_plugin_with_core().first;
}

std::shared_ptr<ov::npuw::CompiledModel> make_compiled_model_with_input_link(
    const std::pair<std::size_t, std::size_t>& input_link) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    param->output(0).get_tensor().set_names({"input"});
    auto res = std::make_shared<ov::op::v0::Result>(param);
    res->output(0).get_tensor().set_names({"output"});
    auto model =
        std::make_shared<ov::Model>(ov::OutputVector{res->output(0)}, ov::ParameterVector{param}, "test_model");

    auto plugin = make_test_plugin();
    auto compiled = std::make_shared<ov::npuw::CompiledModel>(model, plugin, true);

    compiled->m_inputs_to_submodels_inputs = {input_link};
    compiled->m_outputs_to_submodels_outputs = {{0u, 0u}};
    compiled->m_param_subscribers.clear();
    compiled->m_submodels_input_to_prev_output.clear();
    compiled->m_dev_list.clear();
    compiled->m_non_npuw_props.clear();
    compiled->set_weights_bank(ov::npuw::weights::bank("test_bank", plugin->get_core(), ""));

    return compiled;
}

std::shared_ptr<ov::Model> make_simple_model(const std::string& name) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto res = std::make_shared<ov::op::v0::Result>(param);
    return std::make_shared<ov::Model>(ov::OutputVector{res->output(0)}, ov::ParameterVector{param}, name);
}

std::string export_compiled_model(const std::shared_ptr<ov::npuw::CompiledModel>& compiled) {
    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    compiled->export_model(stream);
    return stream.str();
}

std::string make_nested_npuw_orc_chain(std::size_t depth) {
    auto compiled = make_compiled_model_with_input_link(ov::npuw::CompiledModel::NO_LINK);
    compiled->m_dev_list = {"NPU"};
    compiled->m_cfg.update({{"NPUW_DEVICES", "NPU"}});
    auto sub_model = make_simple_model("nested_submodel");
    auto sub_plugin = make_test_plugin();
    ov::npuw::CompiledModel::CompiledModelDesc base_desc;
    base_desc.compiled_model = ov::SoPtr<ov::ICompiledModel>{
        std::make_shared<FakeSubCompiledModel>(sub_model, sub_plugin, std::string{}, "NPU")};
    compiled->m_compiled_submodels.push_back(std::move(base_desc));
    auto blob = export_compiled_model(compiled);

    for (std::size_t level = 1; level < depth; ++level) {
        compiled = make_compiled_model_with_input_link(ov::npuw::CompiledModel::NO_LINK);
        compiled->m_dev_list = {"NPU"};
        compiled->m_cfg.update({{"NPUW_DEVICES", "NPU"}});

        auto sub_model = make_simple_model("nested_submodel");
        auto sub_plugin = make_test_plugin();
        ov::npuw::CompiledModel::CompiledModelDesc desc;
        desc.compiled_model =
            ov::SoPtr<ov::ICompiledModel>{std::make_shared<FakeSubCompiledModel>(sub_model, sub_plugin, blob, "NPU")};
        compiled->m_compiled_submodels.push_back(std::move(desc));
        blob = export_compiled_model(compiled);
    }

    return blob;
}

std::string make_nested_batched_wrapper_chain(std::size_t depth) {
    auto blob = make_nested_npuw_orc_chain(1);
    for (std::size_t level = 0; level < depth; ++level) {
        std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
        ov::npuw::s11n::write_header(stream, NPUW_BATCHED_COMPILED_MODEL_INDICATOR);
        ov::npuw::s11n::write(stream, true);
        ov::npuw::s11n::write(stream, false);
        stream.write(blob.data(), static_cast<std::streamsize>(blob.size()));
        blob = stream.str();
    }
    return blob;
}

std::shared_ptr<ov::npuw::ICompiledModel> import_nested_npuw_blob(const std::string& blob) {
    auto plugin_with_core = make_test_plugin_with_core();
    auto plugin = plugin_with_core.first;
    auto core = plugin_with_core.second;
    ON_CALL(*core, import_model(testing::A<std::istream&>(), testing::A<const std::string&>(), testing::_))
        .WillByDefault(testing::Invoke([&](std::istream& nested_stream,
                                           const std::string&,
                                           const ov::AnyMap& properties) -> ov::SoPtr<ov::ICompiledModel> {
            if (nested_stream.peek() == std::char_traits<char>::eof()) {
                return {};
            }
            auto nested = ov::npuw::ICompiledModel::import_model(nested_stream, plugin, properties);
            if (!nested) {
                return {};
            }
            auto model = make_simple_model("nested_import_result");
            return ov::SoPtr<ov::ICompiledModel>{
                std::make_shared<FakeSubCompiledModel>(model, plugin, std::string{}, "NPU")};
        }));

    std::stringstream stream(blob, std::ios::in | std::ios::out | std::ios::binary);
    auto imported = ov::npuw::ICompiledModel::import_model(stream, plugin, {});
    return imported;
}

void add_fake_submodel(const std::shared_ptr<ov::npuw::CompiledModel>& compiled) {
    auto model = make_simple_model("fake_sub");
    auto plugin = make_test_plugin();
    ov::npuw::CompiledModel::CompiledModelDesc desc;
    desc.compiled_model = ov::SoPtr<ov::ICompiledModel>{std::make_shared<FakeSubCompiledModel>(model, plugin)};
    compiled->m_compiled_submodels.push_back(std::move(desc));
}

void expect_validation_throw_contains(const std::shared_ptr<ov::npuw::CompiledModel>& compiled,
                                      const std::string& expected_substr) {
    try {
        ov::npuw::CompiledModel::validate_import_routing_tables(compiled);
        FAIL() << "Expected ov::Exception containing: " << expected_substr;
    } catch (const ov::Exception& ex) {
        EXPECT_NE(std::string(ex.what()).find(expected_substr), std::string::npos) << ex.what();
    }
}

TEST(CompiledModelOrcImportValidationTest, AcceptsNoLinkInputRouting) {
    auto compiled = make_compiled_model_with_input_link(ov::npuw::CompiledModel::NO_LINK);
    add_fake_submodel(compiled);

    ASSERT_NE(compiled, nullptr);
    EXPECT_NO_THROW(ov::npuw::CompiledModel::validate_import_routing_tables(compiled));
}

TEST(CompiledModelOrcImportValidationTest, RejectsInputRoutingToMissingSubmodel) {
    auto compiled = make_compiled_model_with_input_link({0u, 0u});
    expect_validation_throw_contains(compiled, "m_inputs_to_submodels_inputs[0] input submodel index 0");
}

TEST(CompiledModelOrcImportValidationTest, RejectsInputPortIndexOutOfRange) {
    auto compiled = make_compiled_model_with_input_link({0u, 1u});
    add_fake_submodel(compiled);

    expect_validation_throw_contains(compiled, "m_inputs_to_submodels_inputs[0] input port index 1");
}

TEST(CompiledModelOrcImportValidationTest, RejectsOutputRoutingToMissingSubmodel) {
    auto compiled = make_compiled_model_with_input_link(ov::npuw::CompiledModel::NO_LINK);
    compiled->m_outputs_to_submodels_outputs = {{0u, 0u}};

    expect_validation_throw_contains(compiled, "m_outputs_to_submodels_outputs[0] output submodel index 0");
}

TEST(CompiledModelOrcImportValidationTest, RejectsOutputPortIndexOutOfRange) {
    auto compiled = make_compiled_model_with_input_link(ov::npuw::CompiledModel::NO_LINK);
    add_fake_submodel(compiled);
    compiled->m_outputs_to_submodels_outputs = {{0u, 1u}};

    expect_validation_throw_contains(compiled, "m_outputs_to_submodels_outputs[0] output port index 1");
}

TEST(CompiledModelOrcImportValidationTest, RejectsNoLinkOutputRouting) {
    auto compiled = make_compiled_model_with_input_link(ov::npuw::CompiledModel::NO_LINK);
    add_fake_submodel(compiled);
    compiled->m_outputs_to_submodels_outputs = {ov::npuw::CompiledModel::NO_LINK};

    expect_validation_throw_contains(compiled, "m_outputs_to_submodels_outputs[0] output link: NO_LINK is not allowed");
}

TEST(CompiledModelOrcImportValidationTest, RejectsReplacedByOutOfRange) {
    auto compiled = make_compiled_model_with_input_link(ov::npuw::CompiledModel::NO_LINK);
    compiled->m_compiled_submodels.emplace_back();
    compiled->m_compiled_submodels[0].replaced_by = 1u;

    expect_validation_throw_contains(compiled, "m_compiled_submodels[0].replaced_by index 1");
}

TEST(CompiledModelOrcImportValidationTest, RejectsInputsTableSizeMismatch) {
    auto compiled = make_compiled_model_with_input_link(ov::npuw::CompiledModel::NO_LINK);
    compiled->m_inputs_to_submodels_inputs.clear();

    expect_validation_throw_contains(compiled, "Invalid m_inputs_to_submodels_inputs size 0");
}

TEST(CompiledModelOrcImportValidationTest, RejectsOutputsTableSizeMismatch) {
    auto compiled = make_compiled_model_with_input_link(ov::npuw::CompiledModel::NO_LINK);
    compiled->m_outputs_to_submodels_outputs.clear();

    expect_validation_throw_contains(compiled, "Invalid m_outputs_to_submodels_outputs size 0");
}

TEST(CompiledModelOrcImportValidationTest, RejectsParamSubscribersKeyOutOfRange) {
    auto compiled = make_compiled_model_with_input_link(ov::npuw::CompiledModel::NO_LINK);
    add_fake_submodel(compiled);
    compiled->m_param_subscribers = {{1u, {{0u, 0u}}}};

    expect_validation_throw_contains(compiled, "Invalid m_param_subscribers key 1");
}

TEST(CompiledModelOrcImportValidationTest, RejectsNoLinkParamSubscriberEntry) {
    auto compiled = make_compiled_model_with_input_link(ov::npuw::CompiledModel::NO_LINK);
    add_fake_submodel(compiled);
    compiled->m_param_subscribers = {{0u, {ov::npuw::CompiledModel::NO_LINK}}};

    expect_validation_throw_contains(compiled, "m_param_subscribers[0] input link: NO_LINK is not allowed");
}

TEST(CompiledModelOrcImportValidationTest, RejectsParamSubscribersPortOutOfRange) {
    auto compiled = make_compiled_model_with_input_link(ov::npuw::CompiledModel::NO_LINK);
    add_fake_submodel(compiled);
    compiled->m_param_subscribers = {{0u, {{0u, 1u}}}};

    expect_validation_throw_contains(compiled, "m_param_subscribers[0] input port index 1");
}

TEST(CompiledModelOrcImportValidationTest, AcceptsValidPrevOutputRouting) {
    auto compiled = make_compiled_model_with_input_link(ov::npuw::CompiledModel::NO_LINK);
    add_fake_submodel(compiled);
    add_fake_submodel(compiled);
    compiled->m_submodels_input_to_prev_output = {{{1u, 0u}, {0u, 0u}}};

    EXPECT_NO_THROW(ov::npuw::CompiledModel::validate_import_routing_tables(compiled));
}

TEST(CompiledModelOrcImportValidationTest, RejectsPrevOutputConsumerSubmodelOutOfRange) {
    auto compiled = make_compiled_model_with_input_link(ov::npuw::CompiledModel::NO_LINK);
    add_fake_submodel(compiled);
    compiled->m_submodels_input_to_prev_output = {{{7u, 0u}, {0u, 0u}}};

    expect_validation_throw_contains(compiled, "m_submodels_input_to_prev_output[0] input submodel index 7");
}

TEST(CompiledModelOrcImportValidationTest, RejectsPrevOutputConsumerPortOutOfRange) {
    auto compiled = make_compiled_model_with_input_link(ov::npuw::CompiledModel::NO_LINK);
    add_fake_submodel(compiled);
    compiled->m_submodels_input_to_prev_output = {{{0u, 5u}, {0u, 0u}}};

    expect_validation_throw_contains(compiled, "m_submodels_input_to_prev_output[0] input port index 5");
}

TEST(CompiledModelOrcImportValidationTest, RejectsPrevOutputProducerSubmodelOutOfRange) {
    auto compiled = make_compiled_model_with_input_link(ov::npuw::CompiledModel::NO_LINK);
    add_fake_submodel(compiled);
    compiled->m_submodels_input_to_prev_output = {{{0u, 0u}, {9u, 0u}}};

    expect_validation_throw_contains(compiled, "m_submodels_input_to_prev_output[0] output submodel index 9");
}

TEST(CompiledModelOrcImportValidationTest, RejectsPrevOutputProducerPortOutOfRange) {
    auto compiled = make_compiled_model_with_input_link(ov::npuw::CompiledModel::NO_LINK);
    add_fake_submodel(compiled);
    compiled->m_submodels_input_to_prev_output = {{{0u, 0u}, {0u, 3u}}};

    expect_validation_throw_contains(compiled, "m_submodels_input_to_prev_output[0] output port index 3");
}

TEST(CompiledModelOrcImportValidationTest, RejectsNoLinkPrevOutputConsumer) {
    auto compiled = make_compiled_model_with_input_link(ov::npuw::CompiledModel::NO_LINK);
    add_fake_submodel(compiled);
    compiled->m_submodels_input_to_prev_output = {{ov::npuw::CompiledModel::NO_LINK, {0u, 0u}}};

    expect_validation_throw_contains(compiled,
                                     "m_submodels_input_to_prev_output[0] input link: NO_LINK is not allowed");
}

TEST(CompiledModelOrcImportValidationTest, RejectsNoLinkPrevOutputProducer) {
    auto compiled = make_compiled_model_with_input_link(ov::npuw::CompiledModel::NO_LINK);
    add_fake_submodel(compiled);
    compiled->m_submodels_input_to_prev_output = {{{0u, 0u}, ov::npuw::CompiledModel::NO_LINK}};

    expect_validation_throw_contains(compiled,
                                     "m_submodels_input_to_prev_output[0] output link: NO_LINK is not allowed");
}

TEST(CompiledModelOrcImportValidationTest, AcceptsNestedNpuwOrcImportsBelowDepthLimit) {
    auto imported = import_nested_npuw_blob(make_nested_npuw_orc_chain(4));

    EXPECT_NE(imported, nullptr);
}

TEST(CompiledModelOrcImportValidationTest, RejectsExcessiveNestedNpuwOrcImports) {
    try {
        import_nested_npuw_blob(make_nested_npuw_orc_chain(64));
        FAIL() << "Expected nested NPUW ORC imports to exceed the configured depth limit";
    } catch (const ov::Exception& ex) {
        EXPECT_NE(std::string(ex.what()).find("maximum depth of 32"), std::string::npos) << ex.what();
    }
}

TEST(CompiledModelOrcImportValidationTest, AcceptsNestedBatchedWrapperImportsBelowDepthLimit) {
    auto imported = import_nested_npuw_blob(make_nested_batched_wrapper_chain(4));

    EXPECT_NE(imported, nullptr);
}

TEST(CompiledModelOrcImportValidationTest, RejectsExcessiveNestedBatchedWrapperImports) {
    try {
        import_nested_npuw_blob(make_nested_batched_wrapper_chain(40));
        FAIL() << "Expected nested batched wrappers to exceed the configured import depth limit";
    } catch (const ov::Exception& ex) {
        EXPECT_NE(std::string(ex.what()).find("maximum depth of 32"), std::string::npos) << ex.what();
    }
}

}  // namespace