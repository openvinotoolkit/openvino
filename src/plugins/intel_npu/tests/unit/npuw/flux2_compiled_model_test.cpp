// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "flux2_compiled_model.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "llm_test_helpers.hpp"
#include "openvino/core/version.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "serialization.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_icore.hpp"

namespace {

using ov::test::npuw::MockSubCompiledModel;
using ov::test::npuw::NullPlugin;

// Key strings the inner compile call is expected to carry. Kept as literals so the
// test breaks loudly if an option's public name ever changes.
constexpr const char* kDevices = "NPUW_DEVICES";
constexpr const char* kDynQuant = "NPU_COMPILER_DYNAMIC_QUANTIZATION";
constexpr const char* kF16IC = "NPUW_F16IC";
constexpr const char* kCompileParams = "NPU_COMPILATION_MODE_PARAMS";
constexpr const char* kOnlinePipeline = "NPUW_ONLINE_PIPELINE";
constexpr const char* kTiles = "NPU_TILES";

std::shared_ptr<ov::Model> make_model(const std::string& model_name,
                                      const ov::element::Type& type,
                                      const ov::PartialShape& shape,
                                      const std::string& param_name) {
    auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
    param->set_friendly_name(param_name);
    auto result = std::make_shared<ov::op::v0::Result>(param);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param}, model_name);
}

bool ends_with(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

struct CompileCall {
    ov::AnyMap props;
    std::shared_ptr<ov::Model> model;
};

class RecordingFactory {
public:
    ov::npuw::Flux2CompiledModel::CompiledModelFactory make_factory() {
        return [this](const std::shared_ptr<ov::Model>& model,
                      const std::shared_ptr<const ov::IPlugin>& plugin,
                      const ov::AnyMap& props) -> std::shared_ptr<ov::npuw::ICompiledModel> {
            m_calls.push_back({props, model});
            return std::make_shared<MockSubCompiledModel>(model, plugin, props);
        };
    }

    const CompileCall& only_call() const {
        OPENVINO_ASSERT(m_calls.size() == 1u, "Expected a single compile call");
        return m_calls.front();
    }

private:
    std::vector<CompileCall> m_calls;
};

// Plugin that reports a specific NPU architecture and max-tiles count so the
// architecture-aware tile configuration path can be exercised without real hardware.
class ArchAwarePlugin final : public NullPlugin {
public:
    ArchAwarePlugin(std::string arch, int64_t max_tiles) : m_arch(std::move(arch)), m_max_tiles(max_tiles) {}

    ov::Any get_property(const std::string& name, const ov::AnyMap&) const override {
        if (name == ov::device::architecture.name()) {
            return m_arch;
        }
        if (name == ov::intel_npu::max_tiles.name()) {
            return m_max_tiles;
        }
        return {};
    }

private:
    std::string m_arch;
    int64_t m_max_tiles;
};

std::shared_ptr<ov::MockICore> attach_mock_core_with_npu_device(
    const std::shared_ptr<ov::IPlugin>& plugin,
    std::vector<std::string> device_list = std::vector<std::string>{"0"}) {
    auto core = std::make_shared<testing::NiceMock<ov::MockICore>>();
    plugin->set_core(core);

    ON_CALL(*core,
            get_property(testing::StrEq("NPU"),
                         testing::StrEq(ov::available_devices.name()),
                         testing::An<const ov::AnyMap&>()))
        .WillByDefault([device_list](const std::string&, const std::string&, const ov::AnyMap&) -> ov::Any {
            return ov::Any(device_list);
        });
    return core;
}

// Build a header matching Flux2CompiledModel::export_model so import_model's validation
// can be exercised without a real inner ORC blob.
std::string make_flux2_header(const ov::npuw::s11n::IndicatorType& serialization_indicator,
                              const ov::npuw::s11n::IndicatorType& model_indicator,
                              int vmajor,
                              int vminor,
                              int vpatch,
                              const std::string& s11n_version) {
    std::ostringstream stream;
    ov::npuw::s11n::write(stream, serialization_indicator);
    ov::npuw::s11n::write(stream, model_indicator);
    ov::npuw::s11n::write(stream, vmajor);
    ov::npuw::s11n::write(stream, vminor);
    ov::npuw::s11n::write(stream, vpatch);
    ov::npuw::s11n::write(stream, s11n_version);
    return stream.str();
}

class Flux2CompiledModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        m_plugin = std::make_shared<NullPlugin>();
    }

    static ov::AnyMap base_props() {
        return {{"NPUW_FLUX2", "YES"}};
    }

    static void merge_props(ov::AnyMap& dst, const ov::AnyMap& src) {
        for (const auto& [key, value] : src) {
            dst[key] = value;
        }
    }

    std::unique_ptr<ov::npuw::Flux2CompiledModel> create_compiled_model(const std::shared_ptr<ov::Model>& model,
                                                                        const ov::AnyMap& extra_props,
                                                                        RecordingFactory& recorder) const {
        auto props = base_props();
        merge_props(props, extra_props);
        return std::make_unique<ov::npuw::Flux2CompiledModel>(model, m_plugin, props, recorder.make_factory());
    }

    static void expect_base_defaults(const ov::AnyMap& props) {
        EXPECT_EQ(props.at(kDevices).as<std::string>(), "NPU");
        EXPECT_EQ(props.at(kDynQuant).as<std::string>(), "YES");
    }

    std::shared_ptr<ov::IPlugin> m_plugin;
};

// --- Submodel detection via friendly name -----------------------------------

TEST_F(Flux2CompiledModelTest, TextEncoderByNameDisablesF16IC) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::Flux2CompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(
                        make_model("flux_text_encoder", ov::element::f32, ov::PartialShape{1, 16}, "hidden"),
                        {},
                        recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    expect_base_defaults(call.props);
    EXPECT_EQ(call.props.at(kF16IC).as<std::string>(), "NO");
    EXPECT_EQ(call.props.count(kCompileParams), 0u);
    EXPECT_EQ(call.props.count(kOnlinePipeline), 0u);
}

TEST_F(Flux2CompiledModelTest, TransformerByNameSetsHigherPrecisionMVN) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::Flux2CompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(
                        make_model("flux_transformer", ov::element::f32, ov::PartialShape{1, 16}, "hidden"),
                        {},
                        recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    expect_base_defaults(call.props);
    EXPECT_EQ(call.props.at(kCompileParams).as<std::string>(), "compute-layers-with-higher-precision=MVN");
    EXPECT_EQ(call.props.count(kF16IC), 0u);
    EXPECT_EQ(call.props.count(kOnlinePipeline), 0u);
}

TEST_F(Flux2CompiledModelTest, VaeEncoderByNameDisablesOnlinePipeline) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::Flux2CompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(
                        make_model("flux_vae_encoder", ov::element::f32, ov::PartialShape{1, 16}, "sample"),
                        {},
                        recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    expect_base_defaults(call.props);
    EXPECT_EQ(call.props.at(kOnlinePipeline).as<std::string>(), "NONE");
}

TEST_F(Flux2CompiledModelTest, VaeDecoderByNameDisablesOnlinePipeline) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::Flux2CompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(
                        make_model("flux_vae_decoder", ov::element::f32, ov::PartialShape{1, 16}, "latent"),
                        {},
                        recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    expect_base_defaults(call.props);
    EXPECT_EQ(call.props.at(kOnlinePipeline).as<std::string>(), "NONE");
}

// --- Submodel detection via input signature ---------------------------------

TEST_F(Flux2CompiledModelTest, TransformerDetectedFromEncoderHiddenStatesInput) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::Flux2CompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(
                        make_model("submodel", ov::element::f32, ov::PartialShape{1, 16}, "encoder_hidden_states"),
                        {},
                        recorder));
    ASSERT_NE(compiled, nullptr);

    EXPECT_EQ(recorder.only_call().props.at(kCompileParams).as<std::string>(),
              "compute-layers-with-higher-precision=MVN");
}

TEST_F(Flux2CompiledModelTest, TextEncoderDetectedFromInputIdsInput) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::Flux2CompiledModel> compiled;

    ASSERT_NO_THROW(
        compiled = create_compiled_model(make_model("submodel", ov::element::i64, ov::PartialShape{1, 16}, "input_ids"),
                                         {},
                                         recorder));
    ASSERT_NE(compiled, nullptr);

    EXPECT_EQ(recorder.only_call().props.at(kF16IC).as<std::string>(), "NO");
}

TEST_F(Flux2CompiledModelTest, VaeEncoderDetectedFromThreeChannelInput) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::Flux2CompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(
                        make_model("submodel", ov::element::f32, ov::PartialShape{1, 3, 32, 32}, "sample"),
                        {},
                        recorder));
    ASSERT_NE(compiled, nullptr);

    EXPECT_EQ(recorder.only_call().props.at(kOnlinePipeline).as<std::string>(), "NONE");
}

TEST_F(Flux2CompiledModelTest, VaeDecoderDetectedFromLatentChannelInput) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::Flux2CompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(
                        make_model("submodel", ov::element::f32, ov::PartialShape{1, 4, 32, 32}, "latent"),
                        {},
                        recorder));
    ASSERT_NE(compiled, nullptr);

    EXPECT_EQ(recorder.only_call().props.at(kOnlinePipeline).as<std::string>(), "NONE");
}

// --- Unknown submodel and precedence ----------------------------------------

TEST_F(Flux2CompiledModelTest, UnknownModelGetsOnlyBaseDefaults) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::Flux2CompiledModel> compiled;

    ASSERT_NO_THROW(
        compiled = create_compiled_model(make_model("submodel", ov::element::f32, ov::PartialShape{1, 16}, "sample"),
                                         {},
                                         recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    expect_base_defaults(call.props);
    EXPECT_EQ(call.props.count(kF16IC), 0u);
    EXPECT_EQ(call.props.count(kCompileParams), 0u);
    EXPECT_EQ(call.props.count(kOnlinePipeline), 0u);
}

TEST_F(Flux2CompiledModelTest, UserPropertiesOverrideSubmodelAndBaseDefaults) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::Flux2CompiledModel> compiled;

    // text_encoder would default NPUW_F16IC=NO and base NPU_COMPILER_DYNAMIC_QUANTIZATION=YES;
    // both must yield to the user-provided values.
    ASSERT_NO_THROW(compiled = create_compiled_model(
                        make_model("flux_text_encoder", ov::element::f32, ov::PartialShape{1, 16}, "hidden"),
                        {{kF16IC, "YES"}, {kDynQuant, "NO"}, {kDevices, "CPU"}},
                        recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    EXPECT_EQ(call.props.at(kF16IC).as<std::string>(), "YES");
    EXPECT_EQ(call.props.at(kDynQuant).as<std::string>(), "NO");
    EXPECT_EQ(call.props.at(kDevices).as<std::string>(), "CPU");
}

TEST_F(Flux2CompiledModelTest, AppendsFlux2SuffixToInnerModelName) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::Flux2CompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(
                        make_model("flux_transformer", ov::element::f32, ov::PartialShape{1, 16}, "hidden"),
                        {},
                        recorder));
    ASSERT_NE(compiled, nullptr);

    EXPECT_TRUE(ends_with(recorder.only_call().model->get_friendly_name(), "_flux2"));
}

// --- Serialization ----------------------------------------------------------

TEST_F(Flux2CompiledModelTest, ExportWritesFlux2Indicators) {
    RecordingFactory recorder;
    auto compiled =
        create_compiled_model(make_model("flux_transformer", ov::element::f32, ov::PartialShape{1, 16}, "hidden"),
                              {},
                              recorder);
    ASSERT_NE(compiled, nullptr);

    std::ostringstream out;
    ASSERT_NO_THROW(compiled->export_model(out));

    std::istringstream in(out.str());
    ov::npuw::s11n::IndicatorType serialization_indicator{};
    ov::npuw::s11n::IndicatorType model_indicator{};
    ov::npuw::s11n::read(in, serialization_indicator);
    ov::npuw::s11n::read(in, model_indicator);
    EXPECT_EQ(serialization_indicator, NPUW_SERIALIZATION_INDICATOR);
    EXPECT_EQ(model_indicator, NPUW_FLUX2_COMPILED_MODEL_INDICATOR);
}

TEST_F(Flux2CompiledModelTest, ImportRejectsWrongSerializationIndicator) {
    const auto header = make_flux2_header(NPUW_FLUX2_COMPILED_MODEL_INDICATOR,
                                          NPUW_FLUX2_COMPILED_MODEL_INDICATOR,
                                          OPENVINO_VERSION_MAJOR,
                                          OPENVINO_VERSION_MINOR,
                                          OPENVINO_VERSION_PATCH,
                                          std::string(NPUW_SERIALIZATION_VERSION));
    std::istringstream stream(header);
    EXPECT_THROW(ov::npuw::Flux2CompiledModel::import_model(stream, m_plugin, {}), ov::Exception);
}

TEST_F(Flux2CompiledModelTest, ImportRejectsWrongModelIndicator) {
    const auto header = make_flux2_header(NPUW_SERIALIZATION_INDICATOR,
                                          NPUW_GQA_COMPILED_MODEL_INDICATOR,
                                          OPENVINO_VERSION_MAJOR,
                                          OPENVINO_VERSION_MINOR,
                                          OPENVINO_VERSION_PATCH,
                                          std::string(NPUW_SERIALIZATION_VERSION));
    std::istringstream stream(header);
    EXPECT_THROW(ov::npuw::Flux2CompiledModel::import_model(stream, m_plugin, {}), ov::Exception);
}

TEST_F(Flux2CompiledModelTest, ImportRejectsVersionMismatch) {
    const auto header = make_flux2_header(NPUW_SERIALIZATION_INDICATOR,
                                          NPUW_FLUX2_COMPILED_MODEL_INDICATOR,
                                          OPENVINO_VERSION_MAJOR + 1,
                                          OPENVINO_VERSION_MINOR,
                                          OPENVINO_VERSION_PATCH,
                                          std::string(NPUW_SERIALIZATION_VERSION));
    std::istringstream stream(header);
    EXPECT_THROW(ov::npuw::Flux2CompiledModel::import_model(stream, m_plugin, {}), ov::Exception);
}

// --- Architecture-aware NPU tile configuration ------------------------------

TEST_F(Flux2CompiledModelTest, Arch5000SeriesSetsNpuTilesForTransformer) {
    m_plugin = std::make_shared<ArchAwarePlugin>("5010", 3);
    const auto core = attach_mock_core_with_npu_device(m_plugin);

    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::Flux2CompiledModel> compiled;
    ASSERT_NO_THROW(compiled = create_compiled_model(
                        make_model("flux_transformer", ov::element::f32, ov::PartialShape{1, 16}, "hidden"),
                        {},
                        recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    EXPECT_EQ(call.props.at(kTiles).as<int64_t>(), 3);
    EXPECT_EQ(call.props.at(kCompileParams).as<std::string>(), "compute-layers-with-higher-precision=MVN");
}

TEST_F(Flux2CompiledModelTest, Arch4000AddsOptimizationLevelAndSetsNpuTiles) {
    m_plugin = std::make_shared<ArchAwarePlugin>("4000", 4);
    const auto core = attach_mock_core_with_npu_device(m_plugin);

    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::Flux2CompiledModel> compiled;
    ASSERT_NO_THROW(compiled = create_compiled_model(
                        make_model("flux_transformer", ov::element::f32, ov::PartialShape{1, 16}, "hidden"),
                        {},
                        recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    EXPECT_EQ(call.props.at(kTiles).as<int64_t>(), 4);
    EXPECT_EQ(call.props.at(kCompileParams).as<std::string>(),
              "compute-layers-with-higher-precision=MVN optimization-level=3");
}

TEST_F(Flux2CompiledModelTest, Arch3720SkipsNpuTiles) {
    m_plugin = std::make_shared<ArchAwarePlugin>("3720", 2);
    const auto core = attach_mock_core_with_npu_device(m_plugin);

    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::Flux2CompiledModel> compiled;
    ASSERT_NO_THROW(compiled = create_compiled_model(
                        make_model("flux_transformer", ov::element::f32, ov::PartialShape{1, 16}, "hidden"),
                        {},
                        recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    EXPECT_EQ(call.props.count(kTiles), 0u);
    EXPECT_EQ(call.props.at(kCompileParams).as<std::string>(), "compute-layers-with-higher-precision=MVN");
}

TEST_F(Flux2CompiledModelTest, ArchAutoDetectUsesDefaultConfigAndSkipsNpuTiles) {
    m_plugin = std::make_shared<ArchAwarePlugin>("AUTO_DETECT", 4);
    const auto core = attach_mock_core_with_npu_device(m_plugin);

    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::Flux2CompiledModel> compiled;
    ASSERT_NO_THROW(compiled = create_compiled_model(
                        make_model("flux_transformer", ov::element::f32, ov::PartialShape{1, 16}, "hidden"),
                        {},
                        recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    EXPECT_EQ(call.props.count(kTiles), 0u);
    EXPECT_EQ(call.props.at(kCompileParams).as<std::string>(), "compute-layers-with-higher-precision=MVN");
}

TEST_F(Flux2CompiledModelTest, Arch6010AppendsLatencyHintAndSkipsNpuTiles) {
    m_plugin = std::make_shared<ArchAwarePlugin>("6010", 4);
    const auto core = attach_mock_core_with_npu_device(m_plugin);

    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::Flux2CompiledModel> compiled;
    ASSERT_NO_THROW(compiled = create_compiled_model(
                        make_model("flux_transformer", ov::element::f32, ov::PartialShape{1, 16}, "hidden"),
                        {},
                        recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    EXPECT_EQ(call.props.count(kTiles), 0u);
    EXPECT_EQ(call.props.at(kCompileParams).as<std::string>(),
              "compute-layers-with-higher-precision=MVN performance-hint-override=latency");
}

TEST_F(Flux2CompiledModelTest, ArchAtLeast6000UsesDefaultConfig) {
    m_plugin = std::make_shared<ArchAwarePlugin>("7000", 8);
    const auto core = attach_mock_core_with_npu_device(m_plugin);

    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::Flux2CompiledModel> compiled;
    ASSERT_NO_THROW(compiled = create_compiled_model(
                        make_model("flux_transformer", ov::element::f32, ov::PartialShape{1, 16}, "hidden"),
                        {},
                        recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    EXPECT_EQ(call.props.count(kTiles), 0u);
    EXPECT_EQ(call.props.at(kCompileParams).as<std::string>(), "compute-layers-with-higher-precision=MVN");
}

// Submodels without their own compile params (e.g. text_encoder) still get NPU_TILES
// on 5000-series, but no NPU_COMPILATION_MODE_PARAMS, since 5010 adds no compile token.
TEST_F(Flux2CompiledModelTest, Arch5000SeriesSetsNpuTilesForTextEncoderWithoutCompileParams) {
    m_plugin = std::make_shared<ArchAwarePlugin>("5010", 3);
    const auto core = attach_mock_core_with_npu_device(m_plugin);

    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::Flux2CompiledModel> compiled;
    ASSERT_NO_THROW(compiled = create_compiled_model(
                        make_model("flux_text_encoder", ov::element::f32, ov::PartialShape{1, 16}, "hidden"),
                        {},
                        recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    EXPECT_EQ(call.props.at(kTiles).as<int64_t>(), 3);
    EXPECT_EQ(call.props.at(kF16IC).as<std::string>(), "NO");
    EXPECT_EQ(call.props.count(kCompileParams), 0u);
}

// --- NPU_COMPILATION_MODE_PARAMS is appended, not overwritten ---------------

// Regression: a user-provided NPU_COMPILATION_MODE_PARAMS used to replace the per-submodel
// precision hint. It must now be appended so both tokens survive.
TEST_F(Flux2CompiledModelTest, UserCompilationParamsAppendToSubmodelDefault) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::Flux2CompiledModel> compiled;
    ASSERT_NO_THROW(compiled = create_compiled_model(
                        make_model("flux_transformer", ov::element::f32, ov::PartialShape{1, 16}, "hidden"),
                        {{kCompileParams, "performance-hint-override=latency"}},
                        recorder));
    ASSERT_NE(compiled, nullptr);

    EXPECT_EQ(recorder.only_call().props.at(kCompileParams).as<std::string>(),
              "compute-layers-with-higher-precision=MVN performance-hint-override=latency");
}

TEST_F(Flux2CompiledModelTest, UserCompilationParamsAppendAfterArchParams) {
    m_plugin = std::make_shared<ArchAwarePlugin>("4000", 4);
    const auto core = attach_mock_core_with_npu_device(m_plugin);

    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::Flux2CompiledModel> compiled;
    ASSERT_NO_THROW(compiled = create_compiled_model(
                        make_model("flux_transformer", ov::element::f32, ov::PartialShape{1, 16}, "hidden"),
                        {{kCompileParams, "performance-hint-override=latency"}},
                        recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    EXPECT_EQ(call.props.at(kTiles).as<int64_t>(), 4);
    EXPECT_EQ(call.props.at(kCompileParams).as<std::string>(),
              "compute-layers-with-higher-precision=MVN optimization-level=3 performance-hint-override=latency");
}

TEST_F(Flux2CompiledModelTest, TransformerUserHigherPrecisionLayersAppendToDefault) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::Flux2CompiledModel> compiled;
    ASSERT_NO_THROW(
        compiled = create_compiled_model(
            make_model("flux_transformer", ov::element::f32, ov::PartialShape{1, 16}, "hidden"),
            {{kCompileParams, "compute-layers-with-higher-precision=Sqrt,MVN performance-hint-override=latency"}},
            recorder));
    ASSERT_NE(compiled, nullptr);

    EXPECT_EQ(recorder.only_call().props.at(kCompileParams).as<std::string>(),
              "compute-layers-with-higher-precision=MVN,Sqrt performance-hint-override=latency");
}

TEST_F(Flux2CompiledModelTest, UserCompilationParamsReplaceMatchingArchParam) {
    m_plugin = std::make_shared<ArchAwarePlugin>("4000", 4);
    const auto core = attach_mock_core_with_npu_device(m_plugin);

    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::Flux2CompiledModel> compiled;
    ASSERT_NO_THROW(compiled = create_compiled_model(
                        make_model("flux_transformer", ov::element::f32, ov::PartialShape{1, 16}, "hidden"),
                        {{kCompileParams, "optimization-level=2"}},
                        recorder));
    ASSERT_NE(compiled, nullptr);

    EXPECT_EQ(recorder.only_call().props.at(kCompileParams).as<std::string>(),
              "compute-layers-with-higher-precision=MVN optimization-level=2");
}

// User-provided NPU_TILES takes priority over the arch-computed default.
TEST_F(Flux2CompiledModelTest, UserNpuTilesOverrideArchDefault) {
    m_plugin = std::make_shared<ArchAwarePlugin>("5010", 3);
    const auto core = attach_mock_core_with_npu_device(m_plugin);

    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::Flux2CompiledModel> compiled;
    ASSERT_NO_THROW(compiled = create_compiled_model(
                        make_model("flux_transformer", ov::element::f32, ov::PartialShape{1, 16}, "hidden"),
                        {{kTiles, static_cast<int64_t>(1)}},
                        recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    EXPECT_EQ(call.props.count(kTiles), 1u);
    EXPECT_EQ(call.props.at(kTiles).as<int64_t>(), 1);
}

// No NPU backend => no architecture query => no NPU_TILES, per-submodel defaults intact.
TEST_F(Flux2CompiledModelTest, NoNpuBackendSkipsNpuTiles) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::Flux2CompiledModel> compiled;
    ASSERT_NO_THROW(compiled = create_compiled_model(
                        make_model("flux_transformer", ov::element::f32, ov::PartialShape{1, 16}, "hidden"),
                        {},
                        recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    EXPECT_EQ(call.props.count(kTiles), 0u);
    EXPECT_EQ(call.props.at(kCompileParams).as<std::string>(), "compute-layers-with-higher-precision=MVN");
}

}  // namespace
