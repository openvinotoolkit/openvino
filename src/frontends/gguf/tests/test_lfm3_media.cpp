// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstring>
#include <filesystem>
#include <fstream>

#include "cnpy.h"
#include "common_test_utils/common_utils.hpp"
#include "gtest/gtest.h"
#include "op_test_utils.hpp"
#include "openvino/frontend/gguf/adapt_mmproj_to_genai.hpp"
#include "openvino/frontend/gguf/frontend.hpp"
#include "openvino/frontend/gguf/make_stateful.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/serialize.hpp"

namespace {
using Adapter = ov::frontend::gguf::pass::AdaptMmprojToGenAI;
using ov::frontend::gguf::pass::gguf_recurrent_states_key;
using ov::frontend::gguf::pass::GGUFMakeStateful;

void expect_nmse(const ov::Tensor& actual, const cnpy::NpyArray& expected) {
    ASSERT_EQ(actual.get_size(), expected.num_vals);
    double error = 0, energy = 0;
    for (size_t i = 0; i < actual.get_size(); ++i) {
        const auto reference = expected.data<float>()[i];
        const auto difference = double(actual.data<const float>()[i]) - reference;
        error += difference * difference;
        energy += double(reference) * reference;
    }
    ASSERT_GT(energy, 0.);
    EXPECT_LT(error / energy, 1e-5);
}

ov::Tensor tensor(const cnpy::NpyArray& value, ov::element::Type type = ov::element::f32) {
    ov::Tensor result(type, ov::Shape(value.shape.begin(), value.shape.end()));
    std::memcpy(result.data(), value.data<char>(), result.get_byte_size());
    return result;
}

class GGUFTestLfm3Media : public ::testing::TestWithParam<std::string> {};

TEST_P(GGUFTestLfm3Media, IndependentCpuEncoderAndStates) {
    const auto family = GetParam();
    const auto path = std::filesystem::path(TEST_DATA_DIR) / ("lfm3_media_" + family + ".npz");
    ASSERT_TRUE(std::filesystem::is_regular_file(path)) << "Generate with gen_lfm3_media_fixtures.py";
    const auto arrays = cnpy::npz_load(path.string());
    const std::map<std::string, cnpy::NpyArray> data(arrays.begin(), arrays.end());
    ov_gguf_test::TemporaryGguf fixture(data.at("model"));
    const auto model_path = std::filesystem::path(fixture.path);
    const bool vision = family == "vision";
    const bool streaming = family == "audio_stream";
    ov::frontend::gguf::FrontEnd frontend;
    auto native = frontend.convert(frontend.load(model_path.string()));
    ov::Core core;
    for (const bool adapted : {false, true}) {
        auto model = native->clone();
        if (adapted)
            ASSERT_TRUE(Adapter(vision ? Adapter::Modality::VISION : Adapter::Modality::AUDIO).run_on_model(model));
        const auto pairs = model->has_rt_info(gguf_recurrent_states_key())
                               ? model->get_rt_info<std::vector<std::string>>(gguf_recurrent_states_key())
                               : std::vector<std::string>{};
        auto compiled = core.compile_model(
            model,
            "CPU",
            {{ov::hint::inference_precision.name(), ov::element::f32}, {"DYNAMIC_QUANTIZATION_GROUP_SIZE", 0}});
        auto request = compiled.create_infer_request();
        std::map<std::string, ov::Tensor> states;
        for (size_t i = 0; i < pairs.size(); i += 2) {
            auto zero = ov::Tensor(ov::element::f32, model->input(pairs[i]).get_shape());
            std::memset(zero.data(), 0, zero.get_byte_size());
            states.emplace(pairs[i], zero);
        }
        for (size_t step = 0; step < 2; ++step) {
            const auto key = std::to_string(step) + ".";
            for (const auto& input : compiled.inputs()) {
                const auto name = input.get_any_name();
                const auto state = states.find(name);
                if (state != states.end()) {
                    if (!streaming)
                        std::memset(state->second.data(), 0, state->second.get_byte_size());
                    request.set_tensor(name, state->second);
                    continue;
                }
                const auto short_name = name.substr(name.find('.') == std::string::npos ? 0 : name.find('.') + 1);
                request.set_tensor(
                    name,
                    tensor(data.at(key + short_name), short_name == "positions" ? ov::element::i32 : ov::element::f32));
            }
            request.infer();
            const auto output_name = adapted ? (vision ? "image_features" : "audio_features")
                                             : (vision ? "vision.embeddings" : "audio.embeddings");
            expect_nmse(request.get_tensor(output_name), data.at(key + "embeddings"));
            for (size_t i = 0; i < pairs.size(); i += 2) {
                size_t output_index = 0;
                for (; output_index < model->get_results().size(); ++output_index)
                    if (model->get_results()[output_index]->get_friendly_name() == pairs[i + 1])
                        break;
                ASSERT_LT(output_index, model->get_results().size());
                const auto actual = request.get_output_tensor(output_index);
                const auto slot = pairs[i].substr(std::string("audio.state.").size());
                expect_nmse(actual, data.at(key + "state." + slot));
                actual.copy_to(states.at(pairs[i]));
            }
        }
    }
    if (!vision) {
        auto model = native->clone();
        ASSERT_TRUE(Adapter(Adapter::Modality::AUDIO).run_on_model(model));
        ASSERT_TRUE(GGUFMakeStateful().run_on_model(model));
        EXPECT_EQ(model->get_variables().size(), 7);
        EXPECT_EQ(model->get_results().size(), 1);
        const auto xml = model_path.string() + ".xml";
        ov::serialize(model, xml);
        auto restored = core.read_model(xml);
        EXPECT_EQ(restored->get_variables().size(), model->get_variables().size());
        EXPECT_EQ(restored->get_rt_info<std::string>({"gguf_mmproj", "audio.streaming"}), "true");
        std::filesystem::remove(xml);
        std::filesystem::remove(model_path.string() + ".bin");
    }
}

INSTANTIATE_TEST_SUITE_P(Components,
                         GGUFTestLfm3Media,
                         ::testing::Values("vision", "audio", "audio_stream"),
                         [](const ::testing::TestParamInfo<std::string>& info) {
                             return info.param;
                         });

TEST(GGUFLfm3MediaAdapter, CombinedOmniPreservesIndependentComponentsAndStates) {
    const auto path = std::filesystem::path(TEST_DATA_DIR) / "lfm3_media_omni.npz";
    ASSERT_TRUE(std::filesystem::is_regular_file(path));
    const auto arrays = cnpy::npz_load(path.string());
    const std::map<std::string, cnpy::NpyArray> data(arrays.begin(), arrays.end());
    ov_gguf_test::TemporaryGguf fixture(data.at("model"));
    ov::frontend::gguf::FrontEnd frontend;
    const auto native = frontend.convert(frontend.load(fixture.path));
    ov::Core core;
    const auto compare = [&](const std::shared_ptr<ov::Model>& model,
                             const std::map<std::string, std::string>& outputs,
                             size_t state_count) {
        GGUFMakeStateful().run_on_model(model);
        auto compiled = core.compile_model(model,
                                           "CPU",
                                           {{"INFERENCE_PRECISION_HINT", "f32"},
                                            {"DYNAMIC_QUANTIZATION_GROUP_SIZE", 0},
                                            {"KV_CACHE_PRECISION", "f32"}});
        auto request = compiled.create_infer_request();
        for (const auto& input : compiled.inputs()) {
            const auto name = input.get_any_name();
            const auto source = data.count(name)               ? name
                                : data.count("vision." + name) ? "vision." + name
                                                               : "audio." + name;
            ASSERT_TRUE(data.count(source)) << source;
            request.set_tensor(name, tensor(data.at(source), input.get_element_type()));
        }
        request.infer();
        for (const auto& [output, reference] : outputs) {
            const auto& expected = data.at("expected." + reference);
            if (reference == "vision.embeddings" || reference == "audio.embeddings") {
                expect_nmse(request.get_tensor(output), expected);
            } else {
                ov_gguf_test::expect_near(request.get_tensor(output),
                                          {expected.data<float>(), expected.data<float>() + expected.num_vals});
            }
        }
        const auto states = request.query_state();
        ASSERT_EQ(states.size(), state_count);
        for (const auto& state : states) {
            const auto& expected = data.at("state." + state.get_name());
            auto actual = state.get_state();
            if (state.get_name().find("conv") != std::string::npos && actual.get_size() != expected.num_vals) {
                const auto shape = actual.get_shape();
                const auto channels = shape[shape.size() - 2];
                const auto taps = expected.num_vals / channels;
                const auto stride = actual.get_size() / channels;
                ov::Tensor logical(ov::element::f32, {channels, taps});
                for (size_t channel = 0; channel < channels; ++channel)
                    std::copy_n(actual.data<float>() + channel * stride + stride - taps,
                                taps,
                                logical.data<float>() + channel * taps);
                actual = logical;
            }
            if (state.get_name().rfind("audio.state.", 0) == 0) {
                expect_nmse(actual, expected);
            } else {
                ov_gguf_test::expect_near(actual, {expected.data<float>(), expected.data<float>() + expected.num_vals});
            }
        }
    };
    compare(native->clone(),
            {{"vision.embeddings", "vision.embeddings"},
             {"audio.embeddings", "audio.embeddings"},
             {"audio.depth_logits", "audio.depth_logits"},
             {"audio.feedback", "audio.feedback"},
             {"audio.spectrogram", "audio.spectrogram"}},
            20);
    auto vision = native->clone();
    Adapter vision_adapter(Adapter::Modality::VISION, Adapter::Layout::VISION_ENCODERS);
    ASSERT_TRUE(vision_adapter.run_on_model(vision));
    ASSERT_EQ(vision_adapter.get_vision_models().size(), 1);
    compare(vision, {{"image_features", "vision.embeddings"}}, 0);
    auto audio = native->clone();
    ASSERT_TRUE(Adapter(Adapter::Modality::AUDIO).run_on_model(audio));
    compare(audio, {{"audio_features", "audio.embeddings"}}, 7);
    auto generation = native->clone();
    Adapter generation_adapter(Adapter::Modality::AUDIO_GENERATION);
    ASSERT_TRUE(generation_adapter.run_on_model(generation));
    const auto& models = generation_adapter.get_audio_generation_models();
    compare(models.at("depthformer"), {{"audio.depth_logits", "audio.depth_logits"}}, 0);
    compare(models.at("feedback"), {{"audio.feedback", "audio.feedback"}}, 0);
    compare(models.at("codec"), {{"audio.spectrogram", "audio.spectrogram"}}, 13);
    const auto xml = fixture.path + ".xml";
    ov::serialize(models.at("codec"), xml);
    compare(core.read_model(xml), {{"audio.spectrogram", "audio.spectrogram"}}, 13);
    std::filesystem::remove(xml);
    std::filesystem::remove(fixture.path + ".bin");
}

TEST(GGUFLfm3MediaAdapter, IndependentGenerationMapsPreserveCodecState) {
    auto depth = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 12, 2017});
    depth->set_friendly_name("audio.depth.hidden_states");
    auto feedback = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, 16});
    feedback->set_friendly_name("audio.feedback.codes");
    auto spectrum = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, 1282});
    spectrum->set_friendly_name("audio.codec.codes");
    auto state = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 1, 1});
    state->set_friendly_name("audio.codec.state.pos");
    state->output(0).set_names({"audio.codec.state.pos"});
    auto update = std::make_shared<ov::op::v1::Add>(state, ov::op::v0::Constant::create(ov::element::f32, {}, {1.f}));
    update->output(0).set_names({"codec_position_update"});
    depth->output(0).set_names({"audio.depth_logits"});
    feedback->output(0).set_names({"audio.feedback"});
    spectrum->output(0).set_names({"audio.spectrogram"});
    auto model = std::make_shared<ov::Model>(ov::OutputVector{depth, feedback, spectrum, update},
                                             ov::ParameterVector{depth, feedback, spectrum, state});
    model->get_results().back()->set_friendly_name("codec_position_update");
    model->get_rt_info()["gguf_mmproj"] = ov::AnyMap{{"lfm3", std::string("true")}};
    model->get_rt_info()[gguf_recurrent_states_key()] =
        std::vector<std::string>{"audio.codec.state.pos", "codec_position_update"};
    for (const bool stateful : {false, true}) {
        auto source = model->clone();
        if (stateful)
            ASSERT_TRUE(GGUFMakeStateful().run_on_model(source));
        Adapter adapter(Adapter::Modality::AUDIO_GENERATION);
        ASSERT_TRUE(adapter.run_on_model(source));
        const auto& models = adapter.get_audio_generation_models();
        ASSERT_EQ(models.size(), 3);
        EXPECT_EQ(models.at("depthformer"), source);
        EXPECT_EQ(models.at("depthformer")->get_parameters().size(), 1);
        EXPECT_EQ(models.at("feedback")->get_parameters().size(), 1);
        EXPECT_EQ(models.at("codec")->get_parameters().size(), stateful ? 1 : 2);
        EXPECT_EQ(models.at("codec")->get_sinks().size(), stateful ? 1 : 0);
        EXPECT_EQ(models.at("depthformer")->get_sinks().size(), 0);
        EXPECT_EQ(models.at("feedback")->get_sinks().size(), 0);
    }
}
}  // namespace
