// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <filesystem>

#include "builder/api/metadata_store.hpp"
#include "builder/arch_registry.hpp"
#include "builder/projector_registry.hpp"
#include "cnpy.h"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "gguf_writer.hpp"
#include "gtest/gtest.h"
#include "op_test_utils.hpp"
#include "openvino/frontend/gguf/adapt_mmproj_to_genai.hpp"
#include "openvino/frontend/gguf/extension/projector.hpp"
#include "openvino/frontend/gguf/frontend.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/serialize.hpp"

namespace {
class GGUFMMProj : public ::testing::Test {
protected:
    ov_gguf_test::GgufWriter writer;
    std::string path =
        (std::filesystem::temp_directory_path() / (ov::test::utils::generateTestFilePrefix() + ".gguf")).string();
    void SetUp() override {
        writer.kv_str("general.architecture", "clip");
        writer.kv_bool("clip.use_gelu", true);
    }
    void TearDown() override {
        std::filesystem::remove(path);
        std::filesystem::remove(path + ".xml");
        std::filesystem::remove(path + ".bin");
    }
    void weight(const std::string& name, const std::vector<uint64_t>& dims, bool norm = false) {
        writer.filled_tensor(name, dims, norm, 0.01f);
    }
    void encoder(const std::string& modality, const std::string& projector) {
        const auto meta = "clip." + modality + ".";
        const auto p = modality == "vision" ? std::string("v.") : std::string("a.");
        writer.kv_bool("clip.has_" + modality + "_encoder", true);
        writer.kv_str(meta + "projector_type", projector);
        writer.kv_u32(meta + "embedding_length", 8);
        writer.kv_u32(meta + "feed_forward_length", 12);
        writer.kv_u32(meta + "block_count", 1);
        writer.kv_u32(meta + "projection_dim", 6);
        writer.kv_u32(meta + "attention.head_count", 2);
        writer.kv_f32(meta + "attention.layer_norm_epsilon", 1e-5f);
        weight(p + "position_embd.weight", {8, 16});
        for (const auto* n : {"ln1", "ln2"}) {
            weight(p + "blk.0." + n + ".weight", {8}, true);
            weight(p + "blk.0." + n + ".bias", {8});
        }
        for (const auto* n : {"attn_q", "attn_k", "attn_v", "attn_out"})
            weight(p + "blk.0." + n + ".weight", {8, 8});
        weight(p + "blk.0.ffn_up.weight", {8, 12});
        weight(p + "blk.0.ffn_down.weight", {12, 8});
        if (modality == "vision") {
            writer.kv_u32(meta + "image_size", 8);
            writer.kv_u32(meta + "patch_size", 2);
            writer.kv_u32(meta + "projector.scale_factor", 2);
            weight("v.patch_embd.weight", {2, 2, 3, 8});
            weight("v.patch_embd.bias", {8});
            weight("mm.soft_emb_norm.weight", {8}, true);
            weight("mm.input_projection.weight", {6, 8});
        } else {
            writer.kv_u32(meta + "num_mel_bins", 4);
            weight("a.conv1d.1.weight", {3, 4, 8});
            weight("a.conv1d.2.weight", {3, 8, 8});
            weight("a.conv1d.1.bias", {1, 8});
            weight("a.conv1d.2.bias", {1, 8});
            weight("mm.a.fc.weight", {8, 6});
            weight("mm.a.fc.bias", {6});
        }
    }
    std::shared_ptr<ov::Model> convert() {
        EXPECT_TRUE(writer.write(path));
        ov::frontend::gguf::FrontEnd frontend;
        return frontend.convert(frontend.load(path));
    }
};

TEST_F(GGUFMMProj, ExtensionAddsOneBranchAlongsideBuiltinVision) {
    using namespace ov::frontend::gguf;
    encoder("vision", "gemma3");
    writer.kv_bool("clip.has_audio_encoder", true);
    writer.kv_str("clip.audio.projector_type", "custom-audio");
    ASSERT_TRUE(writer.write(path));
    FrontEnd frontend;
    std::shared_ptr<ArchitectureExtension> extension = std::make_shared<ProjectorExtension>(ProjectorDefinition{
        "clip.audio.custom",
        "clip",
        "audio",
        "custom-audio",
        [](GgufGraphContext& graph) {
            auto input = graph.add_input("audio.custom", ov::element::f32, {1, 1, 2, 3});
            return ProjectorResult{graph.node("GGML_OP_SCALE", {input}, 0, {{"scale", 2.0f}, {"bias", 0.0f}}),
                                   {{"audio.merge", "1"}}};
        },
        {}});
    frontend.add_extension(extension);
    auto model = frontend.convert(frontend.load(path));
    ASSERT_EQ(model->inputs().size(), 2);
    ASSERT_EQ(model->outputs().size(), 2);
    EXPECT_EQ(model->get_rt_info<std::string>({"gguf_mmproj", "audio.projector"}), "custom-audio");
    EXPECT_EQ(model->get_rt_info<std::string>({"gguf_mmproj", "vision.projector"}), "gemma3");
    auto request = ov::Core{}.compile_model(model, "CPU").create_infer_request();
    auto audio = request.get_tensor("audio.custom");
    std::fill_n(audio.data<float>(), audio.get_size(), 3.0f);
    auto vision = request.get_tensor("vision.pixel_values");
    std::fill_n(vision.data<float>(), vision.get_size(), 0.0f);
    request.infer();
    auto output = request.get_tensor("audio.embeddings");
    for (size_t i = 0; i < output.get_size(); ++i)
        EXPECT_FLOAT_EQ(output.data<float>()[i], 6.0f);
    FrontEnd isolated;
    OV_EXPECT_THROW(isolated.convert(isolated.load(path)), ov::Exception, testing::HasSubstr("custom-audio"));
}

TEST_F(GGUFMMProj, ExtensionReusesBuiltinProjectorAndReplacementIsExplicit) {
    using namespace ov::frontend::gguf;
    encoder("vision", "custom-gemma3");
    ASSERT_TRUE(writer.write(path));
    ProjectorDefinition definition{"clip.vision.alias",
                                   "clip",
                                   "vision",
                                   "custom-gemma3",
                                   [](GgufGraphContext& graph) {
                                       return build_builtin_projector(graph, "vision", "gemma3");
                                   },
                                   {}};
    FrontEnd frontend;
    frontend.add_extension(std::make_shared<ProjectorExtension>(definition));
    auto model = frontend.convert(frontend.load(path));
    EXPECT_EQ(model->output().get_shape(), (ov::Shape{1, 1, 4, 6}));
    EXPECT_EQ(model->get_rt_info<std::string>({"gguf_mmproj", "vision.projector"}), "custom-gemma3");
    EXPECT_THROW(frontend.add_extension(std::make_shared<ProjectorExtension>(definition)), ov::Exception);
    definition.build = [](GgufGraphContext& graph) {
        return ProjectorResult{graph.add_input("replacement", ov::element::f32, {1, 1, 2, 3}), {}};
    };
    frontend.add_extension(std::make_shared<ProjectorExtension>(definition, RegistrationMode::Replace));
    EXPECT_EQ(frontend.convert(frontend.load(path))->output().get_shape(), (ov::Shape{1, 1, 2, 3}));
    definition.id = "missing";
    EXPECT_THROW(frontend.add_extension(std::make_shared<ProjectorExtension>(definition, RegistrationMode::Replace)),
                 ov::Exception);
}

TEST_F(GGUFMMProj, ExtensionReplacesOnlyOneBuiltinComponent) {
    using namespace ov::frontend::gguf;
    encoder("vision", "gemma3");
    encoder("audio", "qwen2a");
    ASSERT_TRUE(writer.write(path));
    ProjectorDefinition definition{
        "clip.vision.gemma3",
        "clip",
        "vision",
        "gemma3",
        [](GgufGraphContext& graph) {
            auto input = graph.add_input("replacement", ov::element::f32, {1, 1, 2, 3});
            return ProjectorResult{graph.node("GGML_OP_SCALE", {input}, 0, {{"scale", 2.0f}, {"bias", 0.0f}}), {}};
        },
        {}};
    FrontEnd frontend;
    frontend.add_extension(std::make_shared<ProjectorExtension>(definition, RegistrationMode::Replace));
    auto model = frontend.convert(frontend.load(path));
    EXPECT_EQ(model->output("vision.embeddings").get_shape(), (ov::Shape{1, 1, 2, 3}));
    EXPECT_EQ(model->get_rt_info<std::string>({"gguf_mmproj", "audio.projector"}), "qwen2a");
    EXPECT_EQ(model->inputs().size(), 3);
    EXPECT_NO_THROW(model->input("replacement"));
    EXPECT_NO_THROW(model->input("audio.features"));
    EXPECT_NO_THROW(model->input("audio.position_ids"));
}

TEST_F(GGUFMMProj, LoadedInputRetainsProjectorRegistrationSnapshot) {
    using namespace ov::frontend::gguf;
    encoder("vision", "custom");
    ASSERT_TRUE(writer.write(path));
    ProjectorDefinition definition{"custom",
                                   "clip",
                                   "vision",
                                   "custom",
                                   [](GgufGraphContext& graph) {
                                       return ProjectorResult{graph.add_input("first", ov::element::f32, {1, 1, 2, 3}),
                                                              {}};
                                   },
                                   {}};
    FrontEnd frontend;
    frontend.add_extension(std::make_shared<ProjectorExtension>(definition));
    const auto loaded = frontend.load(path);
    definition.build = [](GgufGraphContext& graph) {
        return ProjectorResult{graph.add_input("second", ov::element::f32, {1, 1, 4, 5}), {}};
    };
    frontend.add_extension(std::make_shared<ProjectorExtension>(definition, RegistrationMode::Replace));
    EXPECT_EQ(frontend.convert(loaded)->output().get_shape(), (ov::Shape{1, 1, 2, 3}));
    EXPECT_EQ(frontend.convert(frontend.load(path))->output().get_shape(), (ov::Shape{1, 1, 4, 5}));
}

TEST(GGUFProjectorRegistry, SeparateListsPredicatesAndAmbiguity) {
    using namespace ov::frontend::gguf;
    ArchRegistry registry;
    const auto architectures = registry.supported_archs();
    const auto projectors = registry.projectors().supported_projectors().size();
    ProjectorDefinition definition{"custom",
                                   "clip",
                                   "vision",
                                   "custom",
                                   [](GgufGraphContext&) {
                                       return ProjectorResult{};
                                   },
                                   {}};
    registry.add_extension(std::make_shared<ProjectorExtension>(definition));
    EXPECT_EQ(registry.supported_archs(), architectures);
    EXPECT_EQ(registry.projectors().supported_projectors().size(), projectors + 1);
    EXPECT_EQ(ArchRegistry{}.projectors().supported_projectors().size(), projectors);
    std::unordered_map<std::string, GGUFMetaData> metadata{{"general.architecture", std::string("clip")}};
    detail::MetadataStore store{metadata};
    GgufMetadata view(store);
    ASSERT_TRUE(registry.projectors().find(view, "vision", "custom"));
    EXPECT_FALSE(registry.projectors().find(view, "audio", "custom"));
    definition.id = "conditional";
    definition.match = [](const GgufMetadata& meta) {
        return meta.has("custom.flag");
    };
    registry.add_extension(std::make_shared<ProjectorExtension>(definition));
    EXPECT_TRUE(registry.projectors().find(view, "vision", "custom"));
    metadata["custom.flag"] = std::string("present");
    EXPECT_THROW(registry.projectors().find(view, "vision", "custom"), ov::Exception);
    metadata["general.architecture"] = std::string("other");
    EXPECT_FALSE(registry.projectors().find(view, "vision", "custom"));
    definition.build = {};
    EXPECT_THROW(registry.add_extension(std::make_shared<ProjectorExtension>(definition)), ov::Exception);
}

TEST_F(GGUFMMProj, EmptyBranchIsRejected) {
    using namespace ov::frontend::gguf;
    encoder("vision", "empty");
    ASSERT_TRUE(writer.write(path));
    FrontEnd frontend;
    frontend.add_extension(std::make_shared<ProjectorExtension>(ProjectorDefinition{"empty",
                                                                                    "clip",
                                                                                    "vision",
                                                                                    "empty",
                                                                                    [](GgufGraphContext&) {
                                                                                        return ProjectorResult{};
                                                                                    },
                                                                                    {}}));
    OV_EXPECT_THROW(frontend.convert(frontend.load(path)), ov::Exception, testing::HasSubstr("returned no output"));
}

TEST_F(GGUFMMProj, Gemma3CompilesAndMetadataSurvivesSerialization) {
    encoder("vision", "gemma3");
    writer.kv_u32_array("clip.test.integer_array", {2, 7, 19});
    writer.kv_str_array("clip.test.string_array", {"a,b", "", "c:d"});
    auto model = convert();
    ASSERT_EQ(model->inputs().size(), 1);
    EXPECT_EQ(model->input().get_any_name(), "vision.pixel_values");
    EXPECT_EQ(model->output().get_shape(), (ov::Shape{1, 1, 4, 6}));
    ov::serialize(model, path + ".xml", path + ".bin");
    ov::Core core;
    auto restored = core.read_model(path + ".xml");
    EXPECT_EQ(restored->get_rt_info<std::string>({"gguf_mmproj", "vision.projector"}), "gemma3");
    EXPECT_EQ(restored->get_rt_info<std::string>({"gguf_mmproj", "clip.test.integer_array"}), "2,7,19");
    EXPECT_EQ(restored->get_rt_info<std::string>({"gguf_mmproj", "clip.test.string_array"}), "3:a,b0:3:c:d");
    EXPECT_EQ(restored->get_rt_info<std::string>({"gguf_mmproj", "clip.test.string_array.encoding"}),
              "length-prefixed-strings");
    auto request =
        core.compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32)).create_infer_request();
    auto input = request.get_input_tensor();
    for (size_t i = 0; i < input.get_size(); ++i)
        input.data<float>()[i] = std::sin(float(i));
    request.infer();
    auto output = request.get_output_tensor();
    double energy = 0;
    for (size_t i = 0; i < output.get_size(); ++i) {
        EXPECT_TRUE(std::isfinite(output.data<float>()[i]));
        energy += std::abs(output.data<float>()[i]);
    }
    EXPECT_GT(energy, 0.01);
}

TEST_F(GGUFMMProj, MixedFileHasIndependentVisionAndAudioBranches) {
    encoder("vision", "gemma3");
    encoder("audio", "qwen2a");
    auto model = convert();
    ASSERT_EQ(model->outputs().size(), 2);
    EXPECT_EQ(model->output(0).get_any_name(), "vision.embeddings");
    EXPECT_EQ(model->output(1).get_any_name(), "audio.embeddings");
    auto vision = model->clone();
    using Adapter = ov::frontend::gguf::pass::AdaptMmprojToGenAI;
    Adapter(Adapter::Modality::VISION).run_on_model(vision);
    EXPECT_EQ(vision->inputs().size(), 1);
    EXPECT_EQ(vision->input().get_any_name(), "pixel_values");
    EXPECT_EQ(vision->output().get_shape(), (ov::Shape{1, 4, 6}));
    ov::Core core;
    auto request =
        core.compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32)).create_infer_request();
    ov::Tensor pixels(ov::element::f32, {1, 3, 8, 8});
    std::fill_n(pixels.data<float>(), pixels.get_size(), 0.5f);
    request.set_tensor("vision.pixel_values", pixels);
    for (size_t frames : {8, 12}) {
        ov::Tensor features(ov::element::f32, {1, 4, 1, frames});
        std::fill_n(features.data<float>(), features.get_size(), 0.25f);
        ov::Tensor ids(ov::element::i32, {1, 1, 1, frames / 2});
        for (size_t i = 0; i < ids.get_size(); ++i)
            ids.data<int32_t>()[i] = int32_t(i);
        request.set_tensor("audio.features", features);
        request.set_tensor("audio.position_ids", ids);
        request.infer();
        EXPECT_EQ(request.get_tensor("audio.embeddings").get_shape(), (ov::Shape{1, 1, frames / 4, 6}));
    }
}

TEST_F(GGUFMMProj, VisionEncoderLayoutRejectsMissingVisionWithoutChangingModel) {
    encoder("audio", "qwen2a");
    auto model = convert();
    ov::frontend::gguf::pass::AdaptVisionEncodersToGenAI adapter;
    OV_EXPECT_THROW(adapter.run_on_model(model), ov::Exception, testing::HasSubstr("no vision.embeddings output"));
    EXPECT_TRUE(adapter.get_vision_models().empty());
    EXPECT_EQ(model->output().get_any_name(), "audio.embeddings");
}

TEST_F(GGUFMMProj, UnsupportedSecondModalityIsNotSilentlyDiscarded) {
    encoder("vision", "gemma3");
    writer.kv_bool("clip.has_audio_encoder", true);
    writer.kv_str("clip.audio.projector_type", "unsupported_audio");
    OV_EXPECT_THROW(convert(), ov::Exception, testing::HasSubstr("unsupported_audio"));
}

TEST_F(GGUFMMProj, LegacyGlobalProjectorTakesPrecedence) {
    encoder("vision", "unused_modality_key");
    writer.kv_str("clip.projector_type", "gemma3");
    EXPECT_NO_THROW(convert());
}

// Legacy Qwen2.5 Omni files name one global projector for both encoders.
TEST_F(GGUFMMProj, Qwen25OmniResolvesProjectorPerModality) {
    encoder("vision", "unused_modality_key");
    encoder("audio", "unused_modality_key");
    writer.kv_str("clip.projector_type", "qwen2.5o");
    writer.kv_u32("clip.vision.spatial_merge_size", 2);
    writer.kv_u32("clip.vision.n_wa_pattern", 2);
    weight("v.patch_embd.weight.1", {2, 2, 3, 8});
    weight("mm.0.weight", {32, 12});
    weight("mm.0.bias", {12});
    weight("mm.2.weight", {12, 6});
    weight("mm.2.bias", {6});
    auto model = convert();
    EXPECT_EQ(model->get_rt_info<std::string>({"gguf_mmproj", "vision.projector"}), "qwen2.5vl_merger");
    EXPECT_EQ(model->get_rt_info<std::string>({"gguf_mmproj", "audio.projector"}), "qwen2a");
    ASSERT_EQ(model->outputs().size(), 2);
    for (const auto* name : {"vision.attention_mask", "vision.output_indices", "audio.features"}) {
        const auto inputs = model->inputs();
        EXPECT_TRUE(std::any_of(inputs.begin(), inputs.end(), [&](const ov::Output<ov::Node>& input) {
            return input.get_names().count(name);
        })) << name;
    }
}

TEST_F(GGUFMMProj, ResamplerRejectsUnknownVersion) {
    encoder("vision", "resampler");
    writer.kv_u32("clip.minicpmv_version", 999);
    OV_EXPECT_THROW(convert(), ov::Exception, testing::HasSubstr("resampler' version 999"));
}

TEST_F(GGUFMMProj, ResamplerRejectsInconsistentQueries) {
    encoder("vision", "resampler");
    writer.kv_u32("clip.minicpmv_query_num", 3);
    weight("resampler.query", {256, 2});
    OV_EXPECT_THROW(convert(), ov::Exception, testing::HasSubstr("query tensor matching query_count"));
}

TEST_F(GGUFMMProj, ResamplerRequiresQueryNormalization) {
    encoder("vision", "resampler");
    weight("resampler.query", {256, 96});
    OV_EXPECT_THROW(convert(), ov::Exception, testing::HasSubstr("resampler.ln_q.weight"));
}

class GGUFMMProjAccuracy : public ::testing::TestWithParam<const char*> {
protected:
    cnpy::npz_t arrays;
    std::shared_ptr<ov::Model> model;
    ov::Core core;
    std::unique_ptr<ov_gguf_test::TemporaryGguf> model_file;

    const cnpy::NpyArray& array(const std::string& name) const {
        return ov_gguf_test::npz_array(arrays, name);
    }
    void SetUp() override {
        arrays = cnpy::npz_load((std::filesystem::path(ov_gguf_test::test_data_dir()) / "mmproj_accuracy" /
                                 (std::string(GetParam()) + ".npz"))
                                    .string());
        model_file = std::make_unique<ov_gguf_test::TemporaryGguf>(array("model"));
        ov::frontend::gguf::FrontEnd frontend;
        model = frontend.convert(frontend.load(model_file->path));
    }
    ov::InferRequest compile(const std::shared_ptr<ov::Model>& current) {
        return core
            .compile_model(current,
                           "CPU",
                           ov::hint::inference_precision(ov::element::f32),
                           ov::hint::dynamic_quantization_group_size(0),
                           ov::inference_num_threads(2))
            .create_infer_request();
    }
    ov::Tensor tensor(const std::string& name, ov::element::Type type) const {
        const auto& values = array(name);
        ov::Tensor result(type, values.shape);
        OPENVINO_ASSERT(result.get_byte_size() == values.num_vals * values.word_size,
                        "Reference tensor type mismatch: ",
                        name);
        std::memcpy(result.data(), values.data<char>(), result.get_byte_size());
        return result;
    }
};

TEST_P(GGUFMMProjAccuracy, EmbeddingsMatchLlamaCPU) {
    if (std::string(GetParam()).find("resampler") == 0) {
        const std::string name = GetParam();
        EXPECT_EQ(model->get_rt_info<std::string>({"gguf_mmproj", "vision.query_count"}),
                  name == "resampler_v2"   ? "96"
                  : name == "resampler_v4" ? "64"
                                           : "3");
        EXPECT_EQ(model->get_rt_info<std::string>({"gguf_mmproj", "vision.minicpmv_version"}),
                  name == "resampler_v2"   ? "2"
                  : name == "resampler_v4" ? "4"
                                           : "3");
    }
    auto request = compile(model);
    const auto& inputs = array("inputs");
    auto data = tensor("inputs", ov::element::f32);
    const bool audio = model->has_rt_info({"gguf_mmproj", "audio.projector"});
    request.set_tensor(audio ? "audio.features" : "vision.pixel_values", data);
    if (audio) {
        const auto frames = inputs.shape.back() / 2;
        ov::Tensor ids(ov::element::i32, {1, 1, 1, frames});
        for (size_t i = 0; i < frames; ++i)
            ids.data<int32_t>()[i] = int32_t(i);
        request.set_tensor("audio.position_ids", ids);
    }
    for (const auto& input : model->inputs()) {
        const auto name = input.get_any_name();
        if (name.rfind("vision.", 0) == 0 && name != "vision.pixel_values") {
            request.set_tensor(name, tensor(name.substr(7), input.get_element_type()));
        }
    }
    request.infer();
    const auto actual = request.get_output_tensor();
    const auto& expected = array("embeddings");
    ASSERT_EQ(actual.get_shape(), expected.shape);
    ov_gguf_test::expect_nmse_below(
        ov_gguf_test::nmse(actual.data<const float>(), expected.data<float>(), actual.get_size()),
        1e-5);

    // Adaptation must preserve every feature, including the channel-packed
    // DeepStack branches, while exposing independently executable modalities.
    auto adapted = model->clone();
    using Adapter = ov::frontend::gguf::pass::AdaptMmprojToGenAI;
    Adapter(audio ? Adapter::Modality::AUDIO : Adapter::Modality::VISION).run_on_model(adapted);
    auto adapted_request = compile(adapted);
    for (const auto& input : adapted->inputs())
        adapted_request.set_tensor(
            input.get_any_name(),
            request.get_tensor(std::string(audio ? "audio." : "vision.") + input.get_any_name()));
    adapted_request.infer();
    const size_t branches = adapted->outputs().size();
    const size_t width = expected.shape.back() / branches;
    for (size_t branch = 0; branch < branches; ++branch) {
        const auto output = adapted_request.get_output_tensor(branch);
        EXPECT_EQ(output.get_shape(), (ov::Shape{expected.shape[1], expected.shape[2], width}));
        ov_gguf_test::Nmse adapted_metric;
        for (size_t i = 0; i < output.get_size(); ++i)
            adapted_metric.add(output.data<const float>()[i],
                               expected.data<float>()[(i / width) * width * branches + branch * width + i % width]);
        ov_gguf_test::expect_nmse_below(adapted_metric, 1e-5);
    }
}

class GGUFMMProjDynamicAccuracy : public GGUFMMProjAccuracy {};

// The low-contrast fixture's patch projection cancels to rounding noise, which differs with the
// platform's summation order (3e-5 on ARM64). Dropping its recentering raises the error to 1e-3.
double dynamic_accuracy_limit(const std::string& family) {
    return family == "gemma4uv_low_contrast" ? 1e-4 : 1e-5;
}

TEST_P(GGUFMMProjDynamicAccuracy, ReusesCompiledModelAcrossGrids) {
    for (bool adapt : {false, true}) {
        auto current = model->clone();
        const bool audio = model->has_rt_info({"gguf_mmproj", "audio.projector"});
        using Adapter = ov::frontend::gguf::pass::AdaptMmprojToGenAI;
        if (adapt)
            Adapter(audio ? Adapter::Modality::AUDIO : Adapter::Modality::VISION).run_on_model(current);
        auto request = compile(current);
        for (int step : {0, 1, 0}) {
            for (const auto& input : current->inputs()) {
                const auto name = input.get_any_name();
                const auto key = std::to_string(step) + "." + (adapt ? name : name.substr(audio ? 6 : 7));
                request.set_tensor(name, tensor(key, input.get_element_type()));
            }
            request.infer();
            const auto output = request.get_output_tensor();
            const auto& expected = array(std::to_string(step) + ".embeddings");
            auto shape = expected.shape;
            if (adapt)
                shape.erase(shape.begin());
            ASSERT_EQ(output.get_shape(), shape);
            ov_gguf_test::expect_nmse_below(
                ov_gguf_test::nmse(output.data<const float>(), expected.data<float>(), output.get_size()),
                dynamic_accuracy_limit(GetParam()),
                std::string(GetParam()) + " step=" + std::to_string(step) + " adapted=" + std::to_string(adapt));
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Reference,
                         GGUFMMProjDynamicAccuracy,
                         ::testing::Values("muse-glimmer",
                                           "qwen2.5vl_merger_grids",
                                           "resampler_grids",
                                           "pixtral",
                                           "pixtral_merge",
                                           "phi4",
                                           "gemma4v",
                                           "gemma4v_one_sided",
                                           "gemma4uv",
                                           "gemma4uv_low_contrast",
                                           "gemma4ua",
                                           "minicpmv4_6",
                                           "gemma4a",
                                           "deepseekocr",
                                           "deepseekocr2",
                                           "deepseekocr_resize",
                                           "deepseekocr_overview",
                                           "deepseekocr2_overview"));

INSTANTIATE_TEST_SUITE_P(Reference,
                         GGUFMMProjAccuracy,
                         ::testing::Values("gemma3",
                                           "gemma3_fused",
                                           "gemma3_legacy",
                                           "idefics3",
                                           "janus_pro",
                                           "qwen2a",
                                           "ultravox",
                                           "voxtral",
                                           "musicflamingo",
                                           "mlp",
                                           "mlp_norm",
                                           "mlp_feature",
                                           "meralion",
                                           "glma",
                                           "qwen2vl_merger",
                                           "qwen2.5vl_merger",
                                           "qwen3vl_merger",
                                           "internvl",
                                           "internvl_qknorm",
                                           "qwen2.5vl_merger_window_video",
                                           "voxtral_odd",
                                           "resampler",
                                           "resampler_v2",
                                           "resampler_v4"));
class GGUFMMProjGenAIVisionLayout : public GGUFMMProjAccuracy {
protected:
    ov::Tensor reference(const std::string& step) {
        auto source = model->clone();
        std::map<std::string, ov::Tensor> inputs;
        for (const auto& input : source->inputs()) {
            const auto name = input.get_any_name().substr(std::string("vision.").size());
            const auto key = step.empty() && name == "pixel_values" ? "inputs" : step + name;
            inputs[input.get_any_name()] = tensor(key, input.get_element_type());
        }
        return infer(source, inputs).front();
    }
    int64_t metadata(const std::string& key) const {
        return std::stoll(model->get_rt_info<std::string>({"gguf_mmproj", key}));
    }
    std::vector<ov::Tensor> infer(const std::shared_ptr<ov::Model>& current,
                                  const std::map<std::string, ov::Tensor>& inputs) {
        auto request = compile(current);
        for (const auto& [name, value] : inputs)
            request.set_tensor(name, value);
        request.infer();
        std::vector<ov::Tensor> outputs;
        for (size_t i = 0; i < current->outputs().size(); ++i) {
            const auto output = request.get_output_tensor(i);
            outputs.emplace_back(output.get_element_type(), output.get_shape());
            output.copy_to(outputs.back());
        }
        return outputs;
    }
};

TEST_P(GGUFMMProjGenAIVisionLayout, MatchesSourceGraph) {
    ov::frontend::gguf::pass::AdaptVisionEncodersToGenAI adapter;
    auto adapted = model->clone();
    ASSERT_TRUE(adapter.run_on_model(adapted));
    const auto& models = adapter.get_vision_models();
    const auto projector = model->get_rt_info<std::string>({"gguf_mmproj", "vision.projector"});
    ASSERT_EQ(models.size(), projector == "qwen3vl_merger" ? 3u : 1u);
    EXPECT_EQ(models.at(projector == "qwen3vl_merger" ? "vision_embeddings_merger" : "vision_embeddings"), adapted);
    const auto patch = size_t(metadata("vision.patch_size"));
    const bool steps = std::any_of(arrays.begin(), arrays.end(), [](const auto& entry) {
        return entry.first == "0.embeddings";
    });
    for (const std::string& step : steps ? std::vector<std::string>{"0.", "1."} : std::vector<std::string>{""}) {
        SCOPED_TRACE(projector + " " + step);
        const auto image = tensor(steps ? step + "pixel_values" : "inputs", ov::element::f32);
        const auto& shape = image.get_shape();
        const size_t height = shape[2], width = shape[3], rows = height / patch, cols = width / patch;
        const size_t count = rows * cols;
        const auto pixel = [&](size_t frame, size_t channel, size_t patch_index, size_t y, size_t x) {
            const size_t row = patch_index / cols * patch + y, col = patch_index % cols * patch + x;
            return image.data<const float>()[((frame * 3 + channel) * height + row) * width + col];
        };
        std::vector<ov::Tensor> actual;
        if (projector == "gemma3") {
            actual = infer(models.at("vision_embeddings"), {{"pixel_values", image}});
        } else if (projector == "gemma4v" || projector == "gemma4uv") {
            const size_t padded = count + 3, dim = patch * patch * 3;
            ov::Tensor values(ov::element::f32, {1, padded, dim});
            ov::Tensor positions(ov::element::i64, {1, padded, 2});
            std::fill_n(values.data<float>(), values.get_size(), 0.f);
            std::fill_n(positions.data<int64_t>(), positions.get_size(), int64_t{-1});
            const auto x = tensor(step + "position_x", ov::element::i32);
            const auto y = tensor(step + "position_y", ov::element::i32);
            for (size_t i = 0; i < count; ++i) {
                positions.data<int64_t>()[2 * i] = x.data<const int32_t>()[i];
                positions.data<int64_t>()[2 * i + 1] = y.data<const int32_t>()[i];
                for (size_t py = 0; py < patch; ++py)
                    for (size_t px = 0; px < patch; ++px)
                        for (size_t c = 0; c < 3; ++c)
                            values.data<float>()[i * dim + (py * patch + px) * 3 + c] = pixel(0, c, i, py, px);
            }
            actual =
                infer(models.at("vision_embeddings"), {{"pixel_values", values}, {"image_position_ids", positions}});
        } else if (projector == "muse-glimmer") {
            const size_t dim = 3 * patch * patch;
            ov::Tensor values(ov::element::f32, {count, dim});
            for (size_t i = 0; i < count; ++i)
                for (size_t c = 0; c < 3; ++c)
                    for (size_t py = 0; py < patch; ++py)
                        for (size_t px = 0; px < patch; ++px)
                            values.data<float>()[i * dim + (c * patch + py) * patch + px] = pixel(0, c, i, py, px);
            ov::Tensor grid(ov::element::i64, {1, 3});
            grid.data<int64_t>()[0] = 1;
            grid.data<int64_t>()[1] = int64_t(rows);
            grid.data<int64_t>()[2] = int64_t(cols);
            actual = infer(models.at("vision_embeddings"), {{"pixel_values", values}, {"image_grid_thw", grid}});
        } else {
            ASSERT_EQ(projector, "qwen3vl_merger");
            const auto order = tensor("patch_indices", ov::element::i32);
            const size_t dim = 6 * patch * patch;
            ov::Tensor hidden(ov::element::f32, {count, dim});
            for (size_t j = 0; j < count; ++j)
                for (size_t c = 0; c < 3; ++c)
                    for (size_t t = 0; t < 2; ++t)
                        for (size_t py = 0; py < patch; ++py)
                            for (size_t px = 0; px < patch; ++px)
                                hidden.data<float>()[j * dim + ((c * 2 + t) * patch + py) * patch + px] =
                                    pixel(t, c, size_t(order.data<const int32_t>()[j]), py, px);
            auto embeddings = infer(models.at("vision_embeddings"), {{"hidden_states", hidden}}).front();
            const size_t embedding_width = embeddings.get_shape()[1];

            const auto& pos_model = models.at("vision_embeddings_pos");
            size_t side = 0;
            for (const auto& node : pos_model->get_ordered_ops()) {
                if (ov::is_type<ov::op::v8::Gather>(node))
                    side = size_t(std::lround(std::sqrt(double(node->get_input_shape(0)[0]))));
            }
            ASSERT_GT(side, 1u);
            ov::Tensor indices(ov::element::i64, {4, count});
            std::vector<float> weights(4 * count);
            for (size_t i = 0; i < count; ++i) {
                const float h = rows > 1 ? float(i / cols) * float(side - 1) / float(rows - 1) : 0.f;
                const float w = cols > 1 ? float(i % cols) * float(side - 1) / float(cols - 1) : 0.f;
                const size_t h0 = size_t(h), w0 = size_t(w);
                const size_t h1 = std::min(h0 + 1, side - 1), w1 = std::min(w0 + 1, side - 1);
                const float dh = h - float(h0), dw = w - float(w0);
                const size_t corners[] = {h0 * side + w0, h0 * side + w1, h1 * side + w0, h1 * side + w1};
                const float corner_weights[] = {(1 - dh) * (1 - dw), (1 - dh) * dw, dh * (1 - dw), dh * dw};
                for (size_t k = 0; k < 4; ++k) {
                    indices.data<int64_t>()[k * count + i] = int64_t(corners[k]);
                    weights[k * count + i] = corner_weights[k];
                }
            }
            const auto table = infer(pos_model, {{"input", indices}}).front();
            for (size_t j = 0; j < count; ++j) {
                const size_t i = size_t(order.data<const int32_t>()[j]);
                for (size_t k = 0; k < 4; ++k)
                    for (size_t d = 0; d < embedding_width; ++d)
                        embeddings.data<float>()[j * embedding_width + d] +=
                            weights[k * count + i] * table.data<const float>()[(k * count + i) * embedding_width + d];
            }

            const auto head = metadata("clip.vision.embedding_length") / metadata("clip.vision.attention.head_count");
            const size_t half = size_t(head / 4);
            const auto ids = tensor("position_ids", ov::element::i32);
            ov::Tensor rotary(ov::element::f32, {count, 2 * half});
            for (size_t j = 0; j < count; ++j)
                for (size_t q = 0; q < half; ++q) {
                    const float frequency = std::pow(10000.f, -float(q) / float(half));
                    rotary.data<float>()[j * 2 * half + q] = float(ids.data<const int32_t>()[j]) * frequency;
                    rotary.data<float>()[j * 2 * half + half + q] =
                        float(ids.data<const int32_t>()[count + j]) * frequency;
                }
            ov::Tensor mask(ov::element::f32, {1, count, count});
            std::fill_n(mask.data<float>(), mask.get_size(), 0.f);
            actual = infer(models.at("vision_embeddings_merger"),
                           {{"hidden_states", embeddings}, {"attention_mask", mask}, {"rotary_pos_emb", rotary}});
        }
        const auto expected = reference(step);
        const auto oracle = tensor(step + "embeddings", ov::element::f32);
        ASSERT_EQ(oracle.get_size(), expected.get_size());
        const size_t tokens = expected.get_shape()[2], packed = expected.get_shape()[3];
        size_t offset = 0;
        for (const auto& output : actual) {
            const size_t levels =
                output.get_shape().size() == 3 && output.get_shape()[0] != 1 ? output.get_shape()[0] : 1;
            const size_t level_width = output.get_shape().back();
            ASSERT_EQ(output.get_size(), levels * tokens * level_width);
            ov_gguf_test::Nmse metric, oracle_metric;
            for (size_t level = 0; level < levels; ++level)
                for (size_t t = 0; t < tokens; ++t)
                    for (size_t d = 0; d < level_width; ++d) {
                        const auto value = output.data<const float>()[(level * tokens + t) * level_width + d];
                        const auto index = t * packed + offset + level * level_width + d;
                        metric.add(value, expected.data<const float>()[index]);
                        oracle_metric.add(value, oracle.data<const float>()[index]);
                    }
            ov_gguf_test::expect_nmse_below(metric, 1e-10);
            ov_gguf_test::expect_nmse_below(oracle_metric, 1e-5);
            offset += levels * level_width;
        }
        EXPECT_EQ(offset, packed);
    }
}

INSTANTIATE_TEST_SUITE_P(
    Reference,
    GGUFMMProjGenAIVisionLayout,
    ::testing::Values("gemma3", "gemma4v", "gemma4v_one_sided", "gemma4uv", "muse-glimmer", "qwen3vl_merger"));
}  // namespace
