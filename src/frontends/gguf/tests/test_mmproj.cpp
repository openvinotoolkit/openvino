// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <filesystem>

#include "cnpy.h"
#include "common_test_utils/common_utils.hpp"
#include "gguf_writer.hpp"
#include "gtest/gtest.h"
#include "op_test_utils.hpp"
#include "openvino/frontend/gguf/adapt_mmproj_to_genai.hpp"
#include "openvino/frontend/gguf/frontend.hpp"
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
        size_t count = 1;
        for (auto d : dims)
            count *= d;
        std::vector<float> values(count);
        for (size_t i = 0; i < count; ++i)
            values[i] = norm ? 1.f + 0.01f * float(i) : 0.1f * std::sin(float(i + 1));
        writer.tensor(name, dims, values);
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
    Adapter(Adapter::Modality::Vision).run_on_model(vision);
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

TEST_F(GGUFMMProj, UnsupportedSecondModalityIsNotSilentlyDiscarded) {
    encoder("vision", "gemma3");
    writer.kv_bool("clip.has_audio_encoder", true);
    writer.kv_str("clip.audio.projector_type", "unsupported_audio");
    try {
        convert();
        FAIL() << "Expected unsupported audio error";
    } catch (const ov::Exception& e) {
        EXPECT_NE(std::string(e.what()).find("unsupported_audio"), std::string::npos);
    }
}

TEST_F(GGUFMMProj, LegacyGlobalProjectorTakesPrecedence) {
    encoder("vision", "unused_modality_key");
    writer.kv_str("clip.projector_type", "gemma3");
    EXPECT_NO_THROW(convert());
}

TEST_F(GGUFMMProj, ResamplerRejectsUnknownVersion) {
    encoder("vision", "resampler");
    writer.kv_u32("clip.minicpmv_version", 999);
    try {
        convert();
        FAIL() << "Expected unsupported resampler version";
    } catch (const ov::Exception& e) {
        EXPECT_NE(std::string(e.what()).find("resampler' version 999"), std::string::npos);
    }
}

TEST_F(GGUFMMProj, ResamplerRejectsInconsistentQueries) {
    encoder("vision", "resampler");
    writer.kv_u32("clip.minicpmv_query_num", 3);
    weight("resampler.query", {256, 2});
    try {
        convert();
        FAIL() << "Expected query-count mismatch";
    } catch (const ov::Exception& e) {
        EXPECT_NE(std::string(e.what()).find("query tensor matching query_count"), std::string::npos);
    }
}

TEST_F(GGUFMMProj, ResamplerRequiresQueryNormalization) {
    encoder("vision", "resampler");
    weight("resampler.query", {256, 96});
    try {
        convert();
        FAIL() << "Expected missing normalization tensor";
    } catch (const ov::Exception& e) {
        EXPECT_NE(std::string(e.what()).find("resampler.ln_q.weight"), std::string::npos);
    }
}

class GGUFMMProjAccuracy : public ::testing::TestWithParam<const char*> {};

TEST_P(GGUFMMProjAccuracy, EmbeddingsMatchLlamaCPU) {
    const auto path =
        (std::filesystem::path(ov_gguf_test::test_data_dir()) / "mmproj_accuracy" / (std::string(GetParam()) + ".npz"))
            .string();
    const auto arrays = cnpy::npz_load(path);
    const auto array = [&](const char* name) -> const cnpy::NpyArray& {
        auto it = std::find_if(arrays.begin(), arrays.end(), [&](const auto& entry) {
            return entry.first == name;
        });
        OPENVINO_ASSERT(it != arrays.end(), "Missing reference array ", name);
        return it->second;
    };
    struct Temporary {
        std::string path =
            (std::filesystem::temp_directory_path() / (ov::test::utils::generateTestFilePrefix() + ".gguf")).string();
        ~Temporary() {
            std::filesystem::remove(path);
        }
    } model_file;
    {
        std::ofstream file(model_file.path, std::ios::binary);
        const auto& bytes = array("model");
        file.write(bytes.data<char>(), bytes.num_vals);
        ASSERT_TRUE(file);
    }
    ov::frontend::gguf::FrontEnd frontend;
    auto model = frontend.convert(frontend.load(model_file.path));
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
    ov::Core core;
    auto request = core.compile_model(model,
                                      "CPU",
                                      ov::hint::inference_precision(ov::element::f32),
                                      ov::hint::dynamic_quantization_group_size(0),
                                      ov::inference_num_threads(2))
                       .create_infer_request();
    const auto& inputs = array("inputs");
    ov::Tensor data(ov::element::f32, inputs.shape);
    std::memcpy(data.data(), inputs.data<float>(), data.get_byte_size());
    const bool audio = std::string(GetParam()) == "qwen2a" || std::string(GetParam()) == "ultravox" ||
                       std::string(GetParam()).find("voxtral") == 0 || std::string(GetParam()) == "musicflamingo" ||
                       std::string(GetParam()) == "meralion" || std::string(GetParam()) == "glma";
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
            const auto& values = array(name.substr(7).c_str());
            ov::Tensor tensor(input.get_element_type(), values.shape);
            std::memcpy(tensor.data(), values.data<char>(), tensor.get_byte_size());
            request.set_tensor(name, tensor);
        }
    }
    request.infer();
    const auto actual = request.get_output_tensor();
    const auto& expected = array("embeddings");
    ASSERT_EQ(actual.get_shape(), expected.shape);
    double error = 0, norm = 0;
    for (size_t i = 0; i < actual.get_size(); ++i) {
        const auto ref = expected.data<float>()[i];
        const auto value = actual.data<const float>()[i];
        ASSERT_TRUE(std::isfinite(value));
        error += double(value - ref) * (value - ref);
        norm += double(ref) * ref;
    }
    ASSERT_GT(norm, 1e-12);
    EXPECT_LT(error / norm, 1e-5);

    // Adaptation must preserve every feature, including the channel-packed
    // DeepStack branches, while exposing independently executable modalities.
    auto adapted = model->clone();
    using Adapter = ov::frontend::gguf::pass::AdaptMmprojToGenAI;
    Adapter(audio ? Adapter::Modality::Audio : Adapter::Modality::Vision).run_on_model(adapted);
    auto adapted_request = core.compile_model(adapted,
                                              "CPU",
                                              ov::hint::inference_precision(ov::element::f32),
                                              ov::hint::dynamic_quantization_group_size(0),
                                              ov::inference_num_threads(2))
                               .create_infer_request();
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
        double adapted_error = 0, adapted_norm = 0;
        for (size_t i = 0; i < output.get_size(); ++i) {
            const float ref = expected.data<float>()[(i / width) * width * branches + branch * width + i % width];
            const double delta = output.data<const float>()[i] - ref;
            adapted_error += delta * delta;
            adapted_norm += double(ref) * ref;
        }
        ASSERT_GT(adapted_norm, 1e-12);
        EXPECT_LT(adapted_error / adapted_norm, 1e-5);
    }
}

class GGUFMMProjDynamicAccuracy : public ::testing::TestWithParam<const char*> {};

TEST_P(GGUFMMProjDynamicAccuracy, ReusesCompiledModelAcrossGrids) {
    const auto path =
        (std::filesystem::path(ov_gguf_test::test_data_dir()) / "mmproj_accuracy" / (std::string(GetParam()) + ".npz"))
            .string();
    const auto arrays = cnpy::npz_load(path);
    const auto array = [&](const std::string& name) -> const cnpy::NpyArray& {
        auto it = std::find_if(arrays.begin(), arrays.end(), [&](const auto& entry) {
            return entry.first == name;
        });
        OPENVINO_ASSERT(it != arrays.end(), "Missing reference array ", name);
        return it->second;
    };
    struct Temporary {
        std::string path =
            (std::filesystem::temp_directory_path() / (ov::test::utils::generateTestFilePrefix() + ".gguf")).string();
        ~Temporary() {
            std::filesystem::remove(path);
        }
    } file;
    {
        std::ofstream stream(file.path, std::ios::binary);
        const auto& bytes = array("model");
        stream.write(bytes.data<char>(), bytes.num_vals);
        ASSERT_TRUE(stream);
    }
    ov::frontend::gguf::FrontEnd frontend;
    auto model = frontend.convert(frontend.load(file.path));
    ov::Core core;
    for (bool adapt : {false, true}) {
        auto current = model->clone();
        const bool audio = std::string(GetParam()).find("gemma4a") == 0 || std::string(GetParam()) == "gemma4ua";
        using Adapter = ov::frontend::gguf::pass::AdaptMmprojToGenAI;
        if (adapt)
            Adapter(audio ? Adapter::Modality::Audio : Adapter::Modality::Vision).run_on_model(current);
        auto request = core.compile_model(current,
                                          "CPU",
                                          ov::hint::inference_precision(ov::element::f32),
                                          ov::hint::dynamic_quantization_group_size(0),
                                          ov::inference_num_threads(2))
                           .create_infer_request();
        for (int step : {0, 1, 0}) {
            for (const auto& input : current->inputs()) {
                const auto name = input.get_any_name();
                const auto& values = array(std::to_string(step) + "." + (adapt ? name : name.substr(audio ? 6 : 7)));
                ov::Tensor tensor(input.get_element_type(), values.shape);
                std::memcpy(tensor.data(), values.data<char>(), tensor.get_byte_size());
                request.set_tensor(name, tensor);
            }
            request.infer();
            const auto output = request.get_output_tensor();
            const auto& expected = array(std::to_string(step) + ".embeddings");
            auto shape = expected.shape;
            if (adapt)
                shape.erase(shape.begin());
            ASSERT_EQ(output.get_shape(), shape);
            double error = 0, energy = 0;
            for (size_t i = 0; i < output.get_size(); ++i) {
                const double ref = expected.data<float>()[i], actual = output.data<float>()[i];
                ASSERT_TRUE(std::isfinite(actual));
                error += (actual - ref) * (actual - ref);
                energy += ref * ref;
            }
            ASSERT_GT(energy, 1e-12);
            EXPECT_LT(error / energy, 1e-5) << GetParam() << " step=" << step << " adapted=" << adapt;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Reference,
                         GGUFMMProjDynamicAccuracy,
                         ::testing::Values("pixtral",
                                           "pixtral_merge",
                                           "phi4",
                                           "gemma4v",
                                           "gemma4uv",
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
                                           "meralion",
                                           "glma",
                                           "qwen2vl_merger",
                                           "qwen2.5vl_merger",
                                           "qwen3vl_merger",
                                           "internvl",
                                           "qwen2.5vl_merger_window_video",
                                           "voxtral_odd",
                                           "resampler",
                                           "resampler_v2",
                                           "resampler_v4"));
}  // namespace
