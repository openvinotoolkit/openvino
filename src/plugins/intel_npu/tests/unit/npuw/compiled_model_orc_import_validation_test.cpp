// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#define private public
#include "compiled_model.hpp"
#undef private

#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
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
    FakeSubCompiledModel(const std::shared_ptr<ov::Model>& model, const std::shared_ptr<const ov::IPlugin>& plugin)
        : ov::ICompiledModel(model, plugin, nullptr, nullptr),
          m_model(model) {}

    void export_model(std::ostream&) const override {}
    std::shared_ptr<const ov::Model> get_runtime_model() const override {
        return m_model;
    }
    void set_property(const ov::AnyMap&) override {}
    ov::Any get_property(const std::string&) const override {
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
};

std::shared_ptr<TestPlugin> make_test_plugin() {
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
    return plugin;
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

}  // namespace