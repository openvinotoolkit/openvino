// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "intel_npu/config/config.hpp"
#include "intel_npu/config/npuw.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/util/codec_xor.hpp"

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
                        std::string execution_device = {})
        : ov::ICompiledModel(model, plugin, nullptr, nullptr),
          m_model(model),
          m_execution_device(std::move(execution_device)) {}

    void export_model(std::ostream&) const override {}
    std::shared_ptr<const ov::Model> get_runtime_model() const override {
        return m_model;
    }
    void set_property(const ov::AnyMap&) override {}
    ov::Any get_property(const std::string& name) const override {
        // Needed by CompiledModel::submodel_device() to resolve which m_dev_list entry this
        // submodel belongs to when a genuine ORC blob is exported in a test.
        if (name == ov::execution_devices.name() && !m_execution_device.empty()) {
            return std::vector<std::string>{m_execution_device};
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
    std::string m_execution_device;
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

// IPlugin::set_core() only keeps a weak_ptr, so callers that need the core to stay alive and
// mockable past this call (e.g. to assert on it later) must use make_test_plugin_with_core().
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

// Builds a genuine ORC blob (via the real export path, not hand-written bytes) with a single
// submodel assigned to `device`. Reused by every test below so that none of them call
// validate_dev_list_against_allowlist() or make_submodel_import_config() directly - they all
// exercise the real deserialize_orc_container() / CompiledModelDesc::serialize() call sites.
// `embedded_allowlist`, when set, forges an NPUW_ALLOWED_IMPORT_DEVICES entry inside the
// blob's own (untrusted) m_non_npuw_props, which is deserialized before the allowlist check
// runs - it must never be able to widen the caller-supplied policy.
std::string build_single_submodel_orc_blob(const std::string& device,
                                           bool encrypted = false,
                                           const std::optional<std::string>& embedded_allowlist = std::nullopt) {
    auto compiled = make_compiled_model_with_input_link(ov::npuw::CompiledModel::NO_LINK);
    compiled->m_dev_list = {device};
    compiled->m_cfg.update({{"NPUW_DEVICES", device}});

    auto sub_model = make_simple_model("fake_sub");
    auto sub_plugin = make_test_plugin();
    ov::npuw::CompiledModel::CompiledModelDesc desc;
    desc.compiled_model =
        ov::SoPtr<ov::ICompiledModel>{std::make_shared<FakeSubCompiledModel>(sub_model, sub_plugin, device)};
    compiled->m_compiled_submodels.push_back(std::move(desc));

    if (embedded_allowlist) {
        compiled->m_non_npuw_props["NPUW_ALLOWED_IMPORT_DEVICES"] = *embedded_allowlist;
    }

    if (encrypted) {
        compiled->m_non_npuw_props[ov::cache_encryption_callbacks.name()] =
            ov::EncryptionCallbacks{ov::util::codec_xor, ov::util::codec_xor};
    }

    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    compiled->export_model(stream);
    return stream.str();
}

// Imports `blob` through the public CompiledModel::import_model() entry point and asserts:
// (1) it is rejected, and (2) the nested-import choke point (the plugin Core's import_model)
// is never reached - this fails both if the check is deleted and if it is moved too late.
void expect_rejected_before_nested_import(const std::string& blob, const std::string& allowed_devices) {
    auto [plugin, core] = make_test_plugin_with_core();
    ASSERT_NE(core, nullptr);
    EXPECT_CALL(*core,
               import_model(testing::A<std::istream&>(), testing::A<const std::string&>(), testing::_))
        .Times(0);

    std::stringstream stream(blob, std::ios::in | std::ios::out | std::ios::binary);
    const ov::AnyMap properties{{"NPU_USE_NPUW", "YES"}, {"NPUW_ALLOWED_IMPORT_DEVICES", allowed_devices}};

    try {
        ov::npuw::CompiledModel::import_model(stream, plugin, properties);
        FAIL() << "Expected ov::Exception for a blob device outside the caller allowlist";
    } catch (const ov::Exception& ex) {
        EXPECT_NE(std::string(ex.what()).find("NPUW_ALLOWED_IMPORT_DEVICES"), std::string::npos) << ex.what();
    }
}

// Imports a genuine ORC `blob` with `properties` and returns how many times the choke-point
// Core::import_model() was actually invoked - 0 means the allowlist check rejected the blob
// before any nested submodel was touched, >=1 means it was let through.
int import_and_count_nested_calls_for_blob(const std::string& blob, const ov::AnyMap& properties) {
    auto [plugin, core] = make_test_plugin_with_core();

    int call_count = 0;
    ON_CALL(*core, import_model(testing::A<std::istream&>(), testing::A<const std::string&>(), testing::_))
        .WillByDefault(testing::Invoke(
            [&](std::istream&, const std::string&, const ov::AnyMap&) -> ov::SoPtr<ov::ICompiledModel> {
                ++call_count;
                return {};
            }));

    std::stringstream stream(blob, std::ios::in | std::ios::out | std::ios::binary);
    try {
        ov::npuw::CompiledModel::import_model(stream, plugin, properties);
    } catch (const ov::Exception&) {
        // Some scenarios legitimately reject before reaching the nested import; the caller
        // only cares whether that nested import was attempted, not whether the whole
        // deserialization eventually succeeds.
    }
    return call_count;
}

// Imports a single-submodel blob assigned to `device` with `properties` and returns how many
// times the choke-point Core::import_model was actually invoked - 0 means the allowlist check
// rejected the blob before any nested submodel was touched, >=1 means it was let through.
int import_and_count_nested_calls(const std::string& device, const ov::AnyMap& properties) {
    return import_and_count_nested_calls_for_blob(build_single_submodel_orc_blob(device), properties);
}

TEST(CompiledModelOrcImportValidationTest, RejectsDisallowedDeviceBeforeNestedImportPlainContainer) {
    expect_rejected_before_nested_import(build_single_submodel_orc_blob("GPU"), "NPU");
}

TEST(CompiledModelOrcImportValidationTest, RejectsDisallowedDeviceBeforeNestedImportEncryptedContainer) {
    expect_rejected_before_nested_import(build_single_submodel_orc_blob("GPU", /*encrypted=*/true), "NPU");
}

// m_non_npuw_props (which can embed NPUW_ALLOWED_IMPORT_DEVICES) is untrusted blob metadata
// deserialized before the allowlist check runs. A blob that embeds its own conflicting
// NPUW_ALLOWED_IMPORT_DEVICES=GPU must not be able to widen the caller's real policy of
// NPUW_ALLOWED_IMPORT_DEVICES=NPU - only the caller-supplied `properties` may authorize a
// device; a future change that merges blob metadata into validation properties would
// silently regress this.
TEST(CompiledModelOrcImportValidationTest, RejectsDisallowedDeviceEvenWhenBlobEmbedsConflictingAllowlist) {
    const auto blob = build_single_submodel_orc_blob("GPU", /*encrypted=*/false, /*embedded_allowlist=*/"GPU");
    const ov::AnyMap properties{{"NPUW_ALLOWED_IMPORT_DEVICES", "NPU"}};

    EXPECT_EQ(import_and_count_nested_calls_for_blob(blob, properties), 0);
}

// An ORC blob's own m_dev_list is untrusted (it comes from the deserialized blob metadata,
// not from the caller), and directly picks which backend Core::import_model() invokes for
// each nested submodel. If the trusted caller opts into NPUW_ALLOWED_IMPORT_DEVICES, any blob
// device outside that allowlist must be rejected before any nested submodel/device is touched.
TEST(CompiledModelOrcImportValidationTest, AcceptsDevListDeviceInsideCallerAllowlist) {
    const ov::AnyMap properties{{"NPUW_ALLOWED_IMPORT_DEVICES", "NPU,CPU"}};
    EXPECT_EQ(import_and_count_nested_calls("NPU", properties), 1);
}

TEST(CompiledModelOrcImportValidationTest, AcceptsAnyDevListWhenNoAllowlistConfigured) {
    // No caller-provided policy: pre-existing behavior for legitimate heterogeneous NPUW
    // caches (e.g. NPUW_DEVICES=NPU,CPU,GPU) must remain unaffected.
    EXPECT_EQ(import_and_count_nested_calls("GPU", ov::AnyMap{}), 1);
}

// An allowlist entry naming a specific device ID authorizes only that exact ID, not other
// IDs of the same device.
TEST(CompiledModelOrcImportValidationTest, AcceptsExactDeviceIdInAllowlist) {
    const ov::AnyMap properties{{"NPUW_ALLOWED_IMPORT_DEVICES", "GPU.0"}};
    EXPECT_EQ(import_and_count_nested_calls("GPU.0", properties), 1);
}

TEST(CompiledModelOrcImportValidationTest, RejectsOtherDeviceIdWhenAllowlistNamesExactId) {
    const ov::AnyMap properties{{"NPUW_ALLOWED_IMPORT_DEVICES", "GPU.0"}};
    EXPECT_EQ(import_and_count_nested_calls("GPU.1", properties), 0);
}

// An ID-less allowlist entry ("GPU") is a wildcard authorizing every ID of that device.
TEST(CompiledModelOrcImportValidationTest, AcceptsDeviceIdWhenAllowlistNamesWildcardDevice) {
    const ov::AnyMap properties{{"NPUW_ALLOWED_IMPORT_DEVICES", "GPU"}};
    EXPECT_EQ(import_and_count_nested_calls("GPU.0", properties), 1);
}

// Captures the `import_config` AnyMap that reaches the real choke point (Core::import_model)
// for a genuine single-submodel blob assigned to `device`, instead of calling
// make_submodel_import_config() directly. `outer_allowlist`, when set, authorizes `device` at
// the outer deserialize_orc_container() gate so execution reaches the per-submodel import path
// being observed here.
ov::AnyMap capture_submodel_import_config(const std::string& device,
                                          const std::optional<std::string>& outer_allowlist) {
    const auto blob = build_single_submodel_orc_blob(device);
    auto [plugin, core] = make_test_plugin_with_core();

    ov::AnyMap captured_config;
    ON_CALL(*core, import_model(testing::A<std::istream&>(), testing::A<const std::string&>(), testing::_))
        .WillByDefault(testing::Invoke(
            [&](std::istream&, const std::string&, const ov::AnyMap& config) -> ov::SoPtr<ov::ICompiledModel> {
                captured_config = config;
                return {};
            }));

    ov::AnyMap properties;
    if (outer_allowlist) {
        properties["NPUW_ALLOWED_IMPORT_DEVICES"] = *outer_allowlist;
    }

    std::stringstream stream(blob, std::ios::in | std::ios::out | std::ios::binary);
    EXPECT_NO_THROW(ov::npuw::CompiledModel::import_model(stream, plugin, properties));

    return captured_config;
}

// CWE-862 follow-up: a submodel's own blob can be a nested NPUW ORC container (e.g.
// attention/MoE submodels), which recurses back into deserialize_orc_container() using this
// very config as its `properties`. The caller's allowlist must be forwarded so that recursive
// import stays governed by the same policy - but only for NPU-targeted submodels, since the
// property is NPU/NPUW-specific and must not reach unrelated plugins.
TEST(CompiledModelOrcImportValidationTest, ForwardsAllowlistToNestedNpuSubmodelImport) {
    const auto config = capture_submodel_import_config("NPU", "NPU");

    const auto it = config.find("NPUW_ALLOWED_IMPORT_DEVICES");
    ASSERT_NE(it, config.end());
    EXPECT_EQ(it->second.as<std::string>(), "NPU");
}

TEST(CompiledModelOrcImportValidationTest, DoesNotForwardAllowlistToNonNpuSubmodelImport) {
    const auto config = capture_submodel_import_config("GPU", "GPU");

    EXPECT_EQ(config.find("NPUW_ALLOWED_IMPORT_DEVICES"), config.end());
}

TEST(CompiledModelOrcImportValidationTest, DoesNotForwardAllowlistWhenCallerDidNotSetOne) {
    const auto config = capture_submodel_import_config("NPU", std::nullopt);

    EXPECT_EQ(config.find("NPUW_ALLOWED_IMPORT_DEVICES"), config.end());
}

}  // namespace