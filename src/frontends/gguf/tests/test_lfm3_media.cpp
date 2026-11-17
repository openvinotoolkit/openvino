// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstring>
#include <filesystem>

#include "cnpy.h"
#include "common_test_utils/common_utils.hpp"
#include "gtest/gtest.h"
#include "model_contract.hpp"
#include "op_test_utils.hpp"
#include "openvino/frontend/gguf/adapt_mmproj_to_genai.hpp"
#include "openvino/frontend/gguf/extension/projector.hpp"
#include "openvino/frontend/gguf/frontend.hpp"
#include "openvino/frontend/gguf/make_stateful.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/serialize.hpp"

namespace {
using Adapter = ov::frontend::gguf::pass::AdaptMmprojToGenAI;
using ov::frontend::gguf::pass::gguf_recurrent_states_key;
using ov::frontend::gguf::pass::GGUFMakeStateful;

void expect_nmse(const ov::Tensor& actual, const cnpy::NpyArray& expected) {
    ASSERT_EQ(actual.get_size(), expected.num_vals);
    ov_gguf_test::expect_nmse_below(
        ov_gguf_test::nmse(actual.data<const float>(), expected.data<float>(), expected.num_vals),
        1e-5);
}

ov::Tensor tensor(const cnpy::NpyArray& value, ov::element::Type type = ov::element::f32) {
    ov::Tensor result(type, ov::Shape(value.shape.begin(), value.shape.end()));
    std::memcpy(result.data(), value.data<char>(), result.get_byte_size());
    return result;
}

using Frames = std::pair<ov::Tensor, ov::Tensor>;

ov::Tensor packed_pixels(const std::vector<Frames>& groups, size_t patch = 16, size_t temporal = 2) {
    OPENVINO_ASSERT(!groups.empty(), "packed_pixels requires at least one group");
    const auto shape = groups.front().first.get_shape();
    OPENVINO_ASSERT(shape.size() == 4 && shape[0] == 1 && shape[1] == 3, "packed_pixels requires [1,3,H,W] frames");
    const auto height = shape[2], width = shape[3];
    OPENVINO_ASSERT(height % patch == 0 && width % patch == 0, "packed_pixels requires patch-aligned frames");
    ov::Tensor packed(ov::element::f32, {groups.size(), height / patch, width / patch, 3 * temporal * patch * patch});
    auto* destination = packed.data<float>();
    for (const auto& frames : groups) {
        OPENVINO_ASSERT(frames.first.get_shape() == shape && frames.second.get_shape() == shape);
        for (size_t row = 0; row < height / patch; ++row)
            for (size_t col = 0; col < width / patch; ++col)
                for (size_t channel = 0; channel < 3; ++channel)
                    for (size_t frame_index = 0; frame_index < temporal; ++frame_index)
                        for (size_t y = 0; y < patch; ++y)
                            for (size_t x = 0; x < patch; ++x)
                                *destination++ = (frame_index == 0 ? frames.first : frames.second)
                                                     .data<const float>()[(channel * height + row * patch + y) * width +
                                                                          col * patch + x];
    }
    return packed;
}

class GGUFTestLfm3Media : public ::testing::TestWithParam<std::string> {};

TEST_P(GGUFTestLfm3Media, IndependentCpuEncoderAndStates) {
    const auto family = GetParam();
    const auto path = std::filesystem::path(ov_gguf_test::test_data_dir()) / ("lfm3_media_" + family + ".npz");
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
        if (adapted) {
            ASSERT_TRUE(Adapter(vision ? Adapter::Modality::VISION : Adapter::Modality::AUDIO).run_on_model(model));
        }
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

TEST(GGUFLfm3MediaAdapter, TemporalPackedLayoutMatchesCpuOracleAcrossGroupsAndIR) {
    const auto path = std::filesystem::path(ov_gguf_test::test_data_dir()) / "lfm3_media_vision.npz";
    ASSERT_TRUE(std::filesystem::is_regular_file(path));
    const auto arrays = cnpy::npz_load(path.string());
    const std::map<std::string, cnpy::NpyArray> data(arrays.begin(), arrays.end());
    ov_gguf_test::TemporaryGguf fixture(data.at("model"));
    ov::frontend::gguf::FrontEnd frontend;
    const auto native = frontend.convert(frontend.load(fixture.path));
    EXPECT_EQ(native->get_rt_info<std::string>({"gguf_mmproj", "vision.genai_layout"}), "temporal_packed_patches");
    EXPECT_EQ(native->get_rt_info<std::string>({"gguf_mmproj", "vision.patch_size"}), "16");
    EXPECT_EQ(native->get_rt_info<std::string>({"gguf_mmproj", "vision.temporal_patch_size"}), "2");
    ov::Core core;
    const ov::AnyMap properties{{"INFERENCE_PRECISION_HINT", "f32"}, {"DYNAMIC_QUANTIZATION_GROUP_SIZE", 0}};
    auto native_request = core.compile_model(native, "CPU", properties).create_infer_request();
    const auto xml = fixture.path + ".xml";
    ov::serialize(native, xml);
    enum class Reload { None, Native, Adapted };
    for (const auto reload : {Reload::None, Reload::Native, Reload::Adapted}) {
        auto model = reload == Reload::Native ? core.read_model(xml) : native->clone();
        Adapter adapter(Adapter::Modality::VISION, Adapter::Layout::VISION_ENCODERS);
        ASSERT_TRUE(adapter.run_on_model(model));
        ASSERT_EQ(adapter.get_vision_models().size(), 1);
        EXPECT_EQ(adapter.get_vision_models().at("vision_embeddings"), model);
        if (reload == Reload::Adapted) {
            ov::serialize(model, xml);
            model = core.read_model(xml);
        }
        ASSERT_EQ(model->inputs().size(), 1);
        EXPECT_EQ(model->input().get_any_name(), "pixel_values");
        EXPECT_EQ(model->input().get_element_type(), ov::element::f32);
        EXPECT_EQ(model->input().get_partial_shape(), (ov::PartialShape{-1, -1, -1, 1536}));
        EXPECT_EQ(model->output().get_any_name(), "last_hidden_state");
        EXPECT_EQ(model->output().get_element_type(), ov::element::f32);
        EXPECT_EQ(model->output().get_partial_shape(), (ov::PartialShape{-1, -1, 12}));
        auto request = core.compile_model(model, "CPU", properties).create_infer_request();
        for (const size_t step : {0, 1, 0}) {
            const auto key = std::to_string(step) + ".";
            const auto first = tensor(data.at(key + "pixel_values"));
            const auto second = tensor(data.at(key + "pixel_values_1"));
            for (const size_t group_count : {1, 3, 1}) {
                std::vector<Frames> groups{{first, second}, {second, first}, {first, first}};
                groups.resize(group_count);
                request.set_tensor("pixel_values", packed_pixels(groups));
                request.infer();
                const auto actual = request.get_tensor("last_hidden_state");
                const auto& expected = data.at(key + "embeddings");
                EXPECT_EQ(actual.get_shape(), (ov::Shape{group_count, expected.shape[1], expected.shape[2]}));
                ov_gguf_test::expect_nmse_below(
                    ov_gguf_test::nmse(actual.data<const float>(), expected.data<float>(), expected.num_vals),
                    1e-5);
                for (size_t group = 1; group < groups.size(); ++group) {
                    native_request.set_tensor("vision.pixel_values", groups[group].first);
                    native_request.set_tensor("vision.pixel_values_1", groups[group].second);
                    native_request.set_tensor("vision.positions", tensor(data.at(key + "positions"), ov::element::i32));
                    native_request.infer();
                    const auto native_features = native_request.get_tensor("vision.embeddings");
                    ov_gguf_test::expect_nmse_below(
                        ov_gguf_test::nmse(actual.data<const float>() + group * expected.num_vals,
                                           native_features.data<const float>(),
                                           expected.num_vals),
                        1e-5);
                }
            }
        }
    }
    std::filesystem::remove(xml);
    std::filesystem::remove(fixture.path + ".bin");
}

TEST(GGUFLfm3MediaAdapter, TemporalPackedLayoutRequiresMatchingFrameCount) {
    const auto path = std::filesystem::path(ov_gguf_test::test_data_dir()) / "lfm3_media_vision.npz";
    ASSERT_TRUE(std::filesystem::is_regular_file(path));
    const auto arrays = cnpy::npz_load(path.string());
    const std::map<std::string, cnpy::NpyArray> data(arrays.begin(), arrays.end());
    ov_gguf_test::TemporaryGguf fixture(data.at("model"));
    ov::frontend::gguf::FrontEnd frontend;
    const auto native = frontend.convert(frontend.load(fixture.path));
    const auto second =
        ov::as_type_ptr<ov::op::v0::Parameter>(native->input("vision.pixel_values_1").get_node_shared_ptr());
    auto zero = std::make_shared<ov::op::v1::Multiply>(native->input("vision.pixel_values"),
                                                       ov::op::v0::Constant::create(ov::element::f32, {}, {0.f}));
    second->output(0).replace(zero);
    native->remove_parameter(second);
    native->validate_nodes_and_infer_types();
    ov::Core core;
    const ov::AnyMap properties{{"INFERENCE_PRECISION_HINT", "f32"}, {"DYNAMIC_QUANTIZATION_GROUP_SIZE", 0}};
    auto native_request = core.compile_model(native, "CPU", properties).create_infer_request();
    for (const int64_t temporal : {1, 2}) {
        auto model = native->clone();
        model->set_rt_info(std::to_string(temporal), {"gguf_mmproj", "vision.temporal_patch_size"});
        if (temporal == 2) {
            const auto parameters = model->get_parameters();
            const auto results = model->get_results();
            Adapter adapter(Adapter::Modality::VISION, Adapter::Layout::VISION_ENCODERS);
            EXPECT_THROW(adapter.run_on_model(model), ov::Exception);
            EXPECT_TRUE(adapter.get_vision_models().empty());
            EXPECT_EQ(model->get_parameters(), parameters);
            EXPECT_EQ(model->get_results(), results);
            continue;
        }
        ASSERT_TRUE(Adapter(Adapter::Modality::VISION, Adapter::Layout::VISION_ENCODERS).run_on_model(model));
        ASSERT_EQ(model->inputs().size(), 1);
        EXPECT_EQ(model->input().get_partial_shape(), (ov::PartialShape{-1, -1, -1, 768 * temporal}));
        auto request = core.compile_model(model, "CPU", properties).create_infer_request();
        for (const size_t step : {0, 1}) {
            const auto key = std::to_string(step) + ".";
            const auto first = tensor(data.at(key + "pixel_values"));
            native_request.set_tensor("vision.pixel_values", first);
            native_request.set_tensor("vision.positions", tensor(data.at(key + "positions"), ov::element::i32));
            native_request.infer();
            for (const auto& second_frame : {tensor(data.at(key + "pixel_values_1")), first}) {
                request.set_tensor("pixel_values", packed_pixels({{first, second_frame}}, 16, temporal));
                request.infer();
                const auto actual = request.get_tensor("last_hidden_state");
                const auto expected = native_request.get_tensor("vision.embeddings");
                ASSERT_EQ(actual.get_size(), expected.get_size());
                ov_gguf_test::expect_nmse_below(
                    ov_gguf_test::nmse(actual.data<const float>(), expected.data<const float>(), expected.get_size()),
                    1e-5);
            }
        }
    }
}

TEST(GGUFLfm3MediaAdapter, CombinedOmniPreservesIndependentComponentsAndStates) {
    const auto path = std::filesystem::path(ov_gguf_test::test_data_dir()) / "lfm3_media_omni.npz";
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
            request.set_tensor(name,
                               name == "pixel_values" && outputs.count("last_hidden_state")
                                   ? packed_pixels({{tensor(data.at("vision.pixel_values")),
                                                     tensor(data.at("vision.pixel_values_1"))}})
                                   : tensor(data.at(source), input.get_element_type()));
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
                actual = ov_gguf_test::trailing_conv_taps(actual, channels, expected.num_vals / channels);
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
    compare(vision, {{"last_hidden_state", "vision.embeddings"}}, 0);
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

ov::frontend::gguf::ProjectorDefinition lfm3_replacement(const std::string& modality,
                                                         ov::frontend::gguf::ProjectorDefinition::BuildFn build) {
    const auto type = modality == "vision" ? "lfm3vl" : "lfm3a";
    return {"clip." + modality + "." + type, "clip", modality, type, std::move(build), {}};
}

TEST(GGUFLfm3MediaAdapter, BuiltinProjectorHelperBuildsLfm3Types) {
    using namespace ov::frontend::gguf;
    for (const std::string modality : {"vision", "audio"}) {
        SCOPED_TRACE(modality);
        const auto path = std::filesystem::path(ov_gguf_test::test_data_dir()) / ("lfm3_media_" + modality + ".npz");
        ASSERT_TRUE(std::filesystem::is_regular_file(path)) << "Generate with gen_lfm3_media_fixtures.py";
        const auto arrays = cnpy::npz_load(path.string());
        const std::map<std::string, cnpy::NpyArray> data(arrays.begin(), arrays.end());
        ov_gguf_test::TemporaryGguf fixture(data.at("model"));
        FrontEnd reference_frontend;
        auto reference = reference_frontend.convert(reference_frontend.load(fixture.path));
        FrontEnd frontend;
        frontend.add_extension(std::make_shared<ProjectorExtension>(
            lfm3_replacement(
                modality,
                [modality](GgufGraphContext& graph) {
                    return build_builtin_projector(graph, modality, modality == "vision" ? "lfm3vl" : "lfm3a");
                }),
            RegistrationMode::Replace));
        auto model = frontend.convert(frontend.load(fixture.path));
        EXPECT_EQ(model->get_ordered_ops().size(), reference->get_ordered_ops().size());
        EXPECT_EQ(model->inputs().size(), reference->inputs().size());
        EXPECT_NO_THROW(model->output(modality + ".embeddings"));
    }
}

TEST(GGUFLfm3MediaAdapter, ProjectorCanOverrideTemporalPackedLayoutWithRecipeOrCallback) {
    using namespace ov::frontend::gguf;
    const auto path = std::filesystem::path(ov_gguf_test::test_data_dir()) / "lfm3_media_vision.npz";
    ASSERT_TRUE(std::filesystem::is_regular_file(path));
    const auto arrays = cnpy::npz_load(path.string());
    const std::map<std::string, cnpy::NpyArray> data(arrays.begin(), arrays.end());
    ov_gguf_test::TemporaryGguf fixture(data.at("model"));
    for (const bool callback : {false, true}) {
        auto definition = lfm3_replacement("vision", [callback](GgufGraphContext& graph) {
            auto result = build_builtin_projector(graph, "vision", "lfm3vl");
            if (!callback)
                result.genai_layout = ProjectorLayout::Image;
            return result;
        });
        definition.genai_layout = ProjectorLayout::Image;
        if (callback) {
            definition.genai_adapter = [](const std::shared_ptr<ov::Model>& model) {
                return ModelAdapterResult{model, {{"vision_embeddings", model}}};
            };
        }
        FrontEnd frontend;
        frontend.add_extension(std::make_shared<ProjectorExtension>(definition, RegistrationMode::Replace));
        auto model = frontend.convert(frontend.load(fixture.path));
        EXPECT_EQ(model->get_rt_info<std::string>({"gguf_mmproj", "vision.genai_layout"}),
                  callback ? "temporal_packed_patches" : "image");
        Adapter adapter(Adapter::Modality::VISION, Adapter::Layout::VISION_ENCODERS);
        ASSERT_TRUE(adapter.run_on_model(model));
        ASSERT_EQ(model->inputs().size(), 3);
        EXPECT_EQ(model->input("pixel_values").get_partial_shape(), (ov::PartialShape{1, 3, -1, -1}));
        EXPECT_NO_THROW(model->input("pixel_values_1"));
        EXPECT_NO_THROW(model->input("positions"));
        EXPECT_EQ(model->output().get_any_name(), "image_features");
        EXPECT_EQ(adapter.get_vision_models().at("vision_embeddings"), model);
    }
}

TEST(GGUFLfm3MediaAdapter, TemporalPackedLayoutRejectsInvalidMetadataWithoutChangingModel) {
    const auto path = std::filesystem::path(ov_gguf_test::test_data_dir()) / "lfm3_media_vision.npz";
    ASSERT_TRUE(std::filesystem::is_regular_file(path));
    const auto arrays = cnpy::npz_load(path.string());
    const std::map<std::string, cnpy::NpyArray> data(arrays.begin(), arrays.end());
    ov_gguf_test::TemporaryGguf fixture(data.at("model"));
    ov::frontend::gguf::FrontEnd frontend;
    const auto native = frontend.convert(frontend.load(fixture.path));
    for (const auto& [key, value] :
         std::vector<std::pair<std::string, std::string>>{{"vision.temporal_patch_size", "0"},
                                                          {"vision.temporal_patch_size", "1"},
                                                          {"vision.temporal_patch_size", "3"},
                                                          {"vision.patch_size", "0"},
                                                          {"vision.patch_size", "9223372036854775807"}}) {
        auto model = native->clone();
        model->set_rt_info(value, {"gguf_mmproj", key});
        const auto inputs = model->get_parameters();
        const auto outputs = model->get_results();
        Adapter adapter(Adapter::Modality::VISION, Adapter::Layout::VISION_ENCODERS);
        EXPECT_THROW(adapter.run_on_model(model), ov::Exception);
        EXPECT_TRUE(adapter.get_vision_models().empty());
        EXPECT_EQ(model->get_parameters(), inputs);
        EXPECT_EQ(model->get_results(), outputs);
        EXPECT_NO_THROW(Adapter(Adapter::Modality::VISION).run_on_model(model));
        EXPECT_EQ(model->output().get_any_name(), "image_features");
    }
}

TEST(GGUFLfm3MediaAdapter, ProjectorSuppliedContractIsPreserved) {
    using namespace ov::frontend::gguf;
    const auto path = std::filesystem::path(ov_gguf_test::test_data_dir()) / "lfm3_media_audio.npz";
    ASSERT_TRUE(std::filesystem::is_regular_file(path)) << "Generate with gen_lfm3_media_fixtures.py";
    const auto arrays = cnpy::npz_load(path.string());
    const std::map<std::string, cnpy::NpyArray> data(arrays.begin(), arrays.end());
    ov_gguf_test::TemporaryGguf fixture(data.at("model"));
    FrontEnd frontend;
    frontend.add_extension(std::make_shared<ProjectorExtension>(
        lfm3_replacement(
            "audio",
            [](GgufGraphContext& graph) {
                ModelContract contract;
                contract.batchable_recurrent = true;
                contract.component_groups["audio"] = {"custom",
                                                      {{"custom", {{"audio.embeddings"}, "audio.custom_state."}}}};
                graph.configure_model_contract(contract);
                auto input = graph.add_input("audio.custom", ov::element::f32, {1, 1, 2, 2});
                return ProjectorResult{graph.node("GGML_OP_SCALE", {input}, 0, {{"scale", 2.f}, {"bias", 0.f}}), {}};
            }),
        RegistrationMode::Replace));
    auto model = frontend.convert(frontend.load(fixture.path));
    const auto contract = read_model_contract(model);
    EXPECT_EQ(contract.batchable_recurrent, std::optional<bool>(true));
    const auto& group = contract.component_groups.at("audio");
    EXPECT_EQ(group.primary, "custom");
    EXPECT_EQ(group.models.size(), 1);
    EXPECT_EQ(group.models.at("custom").state_prefix, "audio.custom_state.");
    EXPECT_EQ(group.models.at("custom").outputs, std::vector<std::string>{"audio.embeddings"});
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
    ov::frontend::gguf::ModelContract contract;
    contract.component_groups["audio_generation"] = {"depthformer",
                                                     {{"depthformer", {{"audio.depth_logits"}, ""}},
                                                      {"feedback", {{"audio.feedback"}, ""}},
                                                      {"codec", {{"audio.spectrogram"}, "audio.codec.state."}}}};
    ov::frontend::gguf::write_model_contract(model, contract);
    model->get_rt_info()[gguf_recurrent_states_key()] =
        std::vector<std::string>{"audio.codec.state.pos", "codec_position_update"};
    for (const bool stateful : {false, true}) {
        auto source = model->clone();
        if (stateful) {
            ASSERT_TRUE(GGUFMakeStateful().run_on_model(source));
        }
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
