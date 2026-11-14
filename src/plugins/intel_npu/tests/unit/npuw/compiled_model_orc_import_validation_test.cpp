// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "attn/attn_subgraph.hpp"
#include "host_flash_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

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
    FakeSubCompiledModel(const std::shared_ptr<ov::Model>& model, const std::shared_ptr<const ov::IPlugin>& plugin)
        : ov::ICompiledModel(model, plugin, nullptr, nullptr),
          m_model(model) {}

    void export_model(std::ostream&) const override {}
    std::shared_ptr<const ov::Model> get_runtime_model() const override {
        return m_model;
    }
    void set_property(const ov::AnyMap&) override {}
    ov::Any get_property(const std::string& name) const override {
        if (name == ov::execution_devices.name()) {
            return std::vector<std::string>{"CPU"};
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
};

std::shared_ptr<TestPlugin> make_test_plugin(std::shared_ptr<testing::NiceMock<ov::MockICore>> core = nullptr) {
    auto plugin = std::make_shared<TestPlugin>();
    if (!core) {
        core = std::make_shared<testing::NiceMock<ov::MockICore>>();
    }

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
    const std::pair<std::size_t, std::size_t>& input_link,
    std::size_t num_inputs = 1u,
    std::shared_ptr<TestPlugin> plugin = nullptr) {
    ov::ParameterVector params;
    for (std::size_t i = 0; i < num_inputs; ++i) {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
        param->output(0).get_tensor().set_names({"input_" + std::to_string(i)});
        params.push_back(std::move(param));
    }
    auto res = std::make_shared<ov::op::v0::Result>(params.front());
    res->output(0).get_tensor().set_names({"output"});
    auto model =
        std::make_shared<ov::Model>(ov::OutputVector{res->output(0)}, params, "test_model");

    if (!plugin) {
        plugin = make_test_plugin();
    }
    auto compiled = std::make_shared<ov::npuw::CompiledModel>(model, plugin, true);

    compiled->m_inputs_to_submodels_inputs.resize(num_inputs, ov::npuw::CompiledModel::NO_LINK);
    compiled->m_inputs_to_submodels_inputs[0] = input_link;
    compiled->m_outputs_to_submodels_outputs = {{0u, 0u}};
    compiled->m_param_subscribers.clear();
    compiled->m_submodels_input_to_prev_output.clear();
    compiled->m_dev_list.clear();
    compiled->m_non_npuw_props.clear();
    compiled->set_weights_bank(ov::npuw::weights::bank("test_bank", plugin->get_core(), ""));

    return compiled;
}

std::shared_ptr<ov::Model> make_simple_model(const std::string& name, std::size_t num_inputs = 1u) {
    ov::ParameterVector params;
    ov::OutputVector results;
    for (std::size_t i = 0u; i < num_inputs; ++i) {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
        params.push_back(param);
        results.push_back(std::make_shared<ov::op::v0::Result>(param)->output(0));
    }
    return std::make_shared<ov::Model>(results, params, name);
}

void add_fake_submodel(const std::shared_ptr<ov::npuw::CompiledModel>& compiled, std::size_t num_inputs = 1u) {
    auto model = make_simple_model("fake_sub", num_inputs);
    auto plugin = make_test_plugin();
    ov::npuw::CompiledModel::CompiledModelDesc desc;
    desc.compiled_model = ov::SoPtr<ov::ICompiledModel>{std::make_shared<FakeSubCompiledModel>(model, plugin)};
    compiled->m_compiled_submodels.push_back(std::move(desc));
}

// Adds a function-call submodel: no compiled model of its own, indices refer to the body.
void add_fake_funcall_submodel(const std::shared_ptr<ov::npuw::CompiledModel>& compiled,
                               std::size_t body_idx,
                               std::size_t param_base,
                               std::size_t closure_size) {
    ov::npuw::CompiledModel::CompiledModelDesc desc;
    desc.replaced_by = body_idx;
    desc.param_base = param_base;
    desc.closure.get().closure.resize(closure_size);
    compiled->m_compiled_submodels.push_back(std::move(desc));
}

// Attaches a spatial descriptor with a single param to the last submodel, mimicking a
// deserialized ov::npuw::compiled::Spatial coming from an ORC blob.
void set_spatial_on_last_submodel(const std::shared_ptr<ov::npuw::CompiledModel>& compiled,
                                  std::size_t param_idx,
                                  std::size_t param_base,
                                  std::size_t tail_size) {
    auto& desc = compiled->m_compiled_submodels.back();
    desc.param_base = param_base;

    ov::npuw::compiled::Spatial spatial;
    spatial.range = 4u;
    spatial.nway = 4u;
    spatial.out_dim = 0u;
    spatial.nway_iters = 1u;
    spatial.tail_size = tail_size;
    spatial.params.push_back(ov::npuw::compiled::Spatial::Param{param_idx, 0u});
    desc.spatial = std::move(spatial);
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

TEST(CompiledModelOrcImportValidationTest, AcceptsValidSpatialParamIdx) {
    auto compiled = make_compiled_model_with_input_link(ov::npuw::CompiledModel::NO_LINK);
    add_fake_submodel(compiled);
    set_spatial_on_last_submodel(compiled, 0u, 1u, 2u);

    EXPECT_NO_THROW(ov::npuw::CompiledModel::validate_import_routing_tables(compiled));
}

TEST(CompiledModelOrcImportValidationTest, RejectsSpatialBlobWithWildParamIdx) {
    auto compiled = make_compiled_model_with_input_link(ov::npuw::CompiledModel::NO_LINK);
    add_fake_submodel(compiled);
    // Shape of a crafted blob: an index far past the port count, paired with a param_base
    // large enough that the spatial IO vectors would be resized to accept it.
    set_spatial_on_last_submodel(compiled, 0xDEADBEEFu, 0xFFFFFFFFu, 2u);

    expect_validation_throw_contains(compiled, "m_compiled_submodels[0].param_base 4294967295");
}

TEST(CompiledModelOrcImportValidationTest, RejectsSpatialParamIdxBeyondParamBase) {
    auto compiled = make_compiled_model_with_input_link(ov::npuw::CompiledModel::NO_LINK);
    add_fake_submodel(compiled);
    set_spatial_on_last_submodel(compiled, 0u, 0u, 2u);

    expect_validation_throw_contains(compiled, "m_compiled_submodels[0].spatial->params[0] input port index");
}

TEST(CompiledModelOrcImportValidationTest, RejectsParamBaseBeyondInputPorts) {
    auto compiled = make_compiled_model_with_input_link(ov::npuw::CompiledModel::NO_LINK);
    add_fake_submodel(compiled);
    compiled->m_compiled_submodels.back().param_base = 2u;

    expect_validation_throw_contains(compiled, "m_compiled_submodels[0].param_base 2");
}

TEST(CompiledModelOrcImportValidationTest, RejectsClosureBeyondBodyInputPorts) {
    auto compiled = make_compiled_model_with_input_link(ov::npuw::CompiledModel::NO_LINK);
    add_fake_submodel(compiled);
    auto& desc = compiled->m_compiled_submodels.back();
    desc.param_base = 1u;  // the body exposes exactly one input port
    desc.closure.get().closure.resize(1u);

    expect_validation_throw_contains(compiled, "m_compiled_submodels[0].param_base 1");
}

TEST(CompiledModelOrcImportValidationTest, RejectsFuncallParamBaseBeyondBodyInputPorts) {
    auto compiled = make_compiled_model_with_input_link(ov::npuw::CompiledModel::NO_LINK);
    add_fake_submodel(compiled);
    add_fake_funcall_submodel(compiled, 0u, 0xDEADBEEFu, 0u);

    expect_validation_throw_contains(compiled, "m_compiled_submodels[1].param_base 3735928559");
}

TEST(CompiledModelOrcImportValidationTest, RejectsFuncallClosureBeyondBodyInputPorts) {
    auto compiled = make_compiled_model_with_input_link(ov::npuw::CompiledModel::NO_LINK);
    add_fake_submodel(compiled);
    add_fake_funcall_submodel(compiled, 0u, 1u, 1u);

    expect_validation_throw_contains(compiled, "m_compiled_submodels[1].param_base 1");
}

TEST(CompiledModelOrcImportValidationTest, AcceptsFuncallParamBaseWithinBodyInputPorts) {
    auto compiled = make_compiled_model_with_input_link(ov::npuw::CompiledModel::NO_LINK);
    add_fake_submodel(compiled);
    add_fake_funcall_submodel(compiled, 0u, 1u, 0u);

    EXPECT_NO_THROW(ov::npuw::CompiledModel::validate_import_routing_tables(compiled));
}

TEST(CompiledModelOrcImportValidationTest, BlockKvHfaLogicalInputsSurviveExportImport) {
    constexpr std::size_t logical_inputs = 8u;
    auto core = std::make_shared<testing::NiceMock<ov::MockICore>>();
    auto plugin = make_test_plugin(core);
    auto compiled = make_compiled_model_with_input_link(ov::npuw::CompiledModel::NO_LINK, logical_inputs, plugin);
    add_fake_submodel(compiled, 7u);

    auto& desc = compiled->m_compiled_submodels.back();
    desc.replaced_by = 0u;
    desc.param_base = logical_inputs;
    ov::npuw::v1::subgraphs::RuntimeBehaviorSpec behavior;
    behavior.handles_function_prologue = true;
    desc.pipeline.runtime_behavior = std::move(behavior);

    auto hfa = std::make_shared<ov::npuw::compiled::HostFlashAttention>();
    auto& info = hfa->_sdpa_attention_info;
    info._query_size = 1u;
    info._context_size = 3u;
    info._k_seq_dim = 0u;
    info._v_seq_dim = 0u;
    info._sdpa_indices = {4u, {0u, 2u}, {1u, 3u}, 5u, 6u, 7u};
    info._tile_input_indices = {5u, 3u, 4u, 6u, 0u, 1u, 2u};
    info._tile_output_indices = {0u, 1u, 2u};
    hfa->_past_tile_size = 1;
    hfa->_final_tile_size = 1;
    hfa->_compiled_tile_model = desc.compiled_model;
    hfa->_compiled_final_tile_model = desc.compiled_model;
    ov::npuw::attn::put_compiled_hfa(desc.pipeline.context, std::move(hfa));

    for (std::size_t i = 0; i < logical_inputs; ++i) {
        compiled->m_inputs_to_submodels_inputs[i] = {0u, i};
    }
    add_fake_funcall_submodel(compiled, 0u, logical_inputs, 0u);
    compiled->m_dev_list = {"CPU"};
    compiled->m_cfg.update({{"NPUW_DEVICES", "CPU"}});

    std::weak_ptr<const ov::IPlugin> weak_plugin = plugin;
    ON_CALL(*core, import_model(testing::An<std::istream&>(), testing::An<const std::string&>(), testing::_))
        .WillByDefault([weak_plugin](std::istream&, const std::string&, const ov::AnyMap&) {
            return ov::SoPtr<ov::ICompiledModel>{
                std::make_shared<FakeSubCompiledModel>(make_simple_model("imported_tile", 7u), weak_plugin.lock())};
        });

    std::stringstream blob;
    compiled->export_model(blob);
    auto imported = ov::npuw::CompiledModel::import_model(blob, plugin, {});

    ASSERT_EQ(imported->m_compiled_submodels.size(), 2u);
    EXPECT_EQ(imported->m_compiled_submodels[0].param_base, logical_inputs);
    EXPECT_EQ(imported->m_compiled_submodels[1].param_base, logical_inputs);
    const auto* imported_hfa = ov::npuw::attn::get_compiled_hfa(imported->m_compiled_submodels[0].pipeline.context);
    ASSERT_NE(imported_hfa, nullptr);
    EXPECT_EQ(imported_hfa->_sdpa_attention_info._sdpa_indices.past_key_blocks, (std::vector<std::size_t>{0u, 2u}));
    EXPECT_EQ(imported_hfa->_sdpa_attention_info._sdpa_indices.past_value_blocks, (std::vector<std::size_t>{1u, 3u}));
    EXPECT_TRUE(imported_hfa->is_valid());
}

// The port vector read at the paired site stays in range (idx < inputs()), so only the
// idx < param_base bound rejects the out-of-range write into the param_base-sized tail vector.
TEST(CompiledModelOrcImportValidationTest, RejectsSpatialParamIdxInClosureRange) {
    auto compiled = make_compiled_model_with_input_link(ov::npuw::CompiledModel::NO_LINK);
    add_fake_submodel(compiled, 2u);
    auto& desc = compiled->m_compiled_submodels.back();
    desc.closure.get().closure.resize(1u);
    set_spatial_on_last_submodel(compiled, 1u, 1u, 2u);

    expect_validation_throw_contains(compiled, "m_compiled_submodels[0].spatial->params[0] input port index 1");
}

// param_base is inflated past the real port count so that the param_base-sized vectors accept
// an idx that is still out of range for the port vector itself.
TEST(CompiledModelOrcImportValidationTest, RejectsInflatedParamBaseWithParamIdxAtPortCount) {
    auto compiled = make_compiled_model_with_input_link(ov::npuw::CompiledModel::NO_LINK);
    add_fake_submodel(compiled, 2u);
    set_spatial_on_last_submodel(compiled, 2u, 3u, 2u);

    expect_validation_throw_contains(compiled, "m_compiled_submodels[0].param_base 3");
}

}  // namespace