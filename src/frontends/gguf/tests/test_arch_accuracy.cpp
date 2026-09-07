// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "cnpy.h"
#include "common_test_utils/common_utils.hpp"
#include "gtest/gtest.h"
#include "op_test_utils.hpp"
#include "openvino/frontend/extension/decoder_transformation.hpp"
#include "openvino/frontend/gguf/adapt_to_genai.hpp"
#include "openvino/frontend/gguf/frontend.hpp"
#include "openvino/frontend/gguf/make_stateful.hpp"
#include "openvino/openvino.hpp"

namespace {
class GGUFArchitectureAccuracy : public ::testing::TestWithParam<const char*> {};

TEST_P(GGUFArchitectureAccuracy, PrefillAndCachedDecodeMatchLlamaCPU) {
    const char* override_dir = std::getenv("OV_GGUF_ACCURACY_DATA");
    const auto directory = override_dir ? std::filesystem::path(override_dir)
                                        : std::filesystem::path(ov_gguf_test::test_data_dir()) / "arch_accuracy";
    ASSERT_TRUE(std::filesystem::exists(directory)) << "Missing architecture reference data: " << directory;
    const auto base = directory / GetParam();
    std::vector<std::vector<int64_t>> schedule{{1, 2, 3}, {4}, {5, 6}};
    std::vector<float> reference;
    int32_t vocab = 0;
    std::string model_path = base.string() + ".gguf";
    struct TemporaryModel {
        std::string path;
        ~TemporaryModel() {
            if (!path.empty())
                std::filesystem::remove(path);
        }
    } temporary;
    if (override_dir) {
        std::ifstream file(base.string() + ".bin", std::ios::binary);
        ASSERT_TRUE(file) << base;
        file.read(reinterpret_cast<char*>(&vocab), sizeof(vocab));
        ASSERT_GT(vocab, 6);
        ASSERT_LT(vocab, 1000000);
        std::ifstream token_file(base.string() + ".bin.tokens");
        if (token_file) {
            schedule.clear();
            size_t count = 0;
            while (token_file >> count) {
                ASSERT_GT(count, 0);
                ASSERT_LE(count, 64);
                std::vector<int64_t> ids(count);
                for (auto& token : ids)
                    ASSERT_TRUE(token_file >> token);
                schedule.push_back(std::move(ids));
            }
        }
        ASSERT_FALSE(schedule.empty());
        reference.resize(schedule.size() * vocab);
        file.read(reinterpret_cast<char*>(reference.data()), reference.size() * sizeof(float));
        ASSERT_TRUE(file);
    } else {
        const auto arrays = cnpy::npz_load(base.string() + ".npz");
        const auto array = [&](const std::string& name) -> const cnpy::NpyArray& {
            const auto it = std::find_if(arrays.begin(), arrays.end(), [&](const auto& entry) {
                return entry.first == name;
            });
            OPENVINO_ASSERT(it != arrays.end(), "Missing reference array ", name);
            return it->second;
        };
        const auto& logits = array("logits");
        ASSERT_EQ(logits.shape.size(), 2);
        ASSERT_EQ(logits.shape[0], 3);
        vocab = static_cast<int32_t>(logits.shape[1]);
        const auto* values = logits.data<float>();
        reference.assign(values, values + 3 * vocab);
        temporary.path =
            (std::filesystem::temp_directory_path() / (ov::test::utils::generateTestFilePrefix() + ".gguf")).string();
        model_path = temporary.path;
        std::ofstream file(model_path, std::ios::binary);
        const auto& bytes = array("model");
        file.write(bytes.data<char>(), bytes.num_vals);
        ASSERT_TRUE(file);
    }
    ov::frontend::gguf::FrontEnd fe;
    fe.add_extension(
        std::make_shared<ov::frontend::DecoderTransformationExtension>(ov::frontend::gguf::pass::GGUFMakeStateful()));
    fe.add_extension(
        std::make_shared<ov::frontend::DecoderTransformationExtension>(ov::frontend::gguf::pass::AdaptToGenAI()));
    auto model = fe.convert(fe.load(model_path));
    ov::Core core;
    auto compiled = core.compile_model(model,
                                       "CPU",
                                       ov::hint::inference_precision(ov::element::f32),
                                       ov::num_streams(1),
                                       ov::inference_num_threads(4),
                                       ov::hint::dynamic_quantization_group_size(0),
                                       ov::hint::kv_cache_precision(ov::element::f16));
    auto request = compiled.create_infer_request();
    size_t past = 0;
    size_t step = 0;
    size_t matching_tokens = 0;
    for (const auto& tokens : schedule) {
        const auto count = tokens.size();
        SCOPED_TRACE("past=" + std::to_string(past) + ", tokens=" + std::to_string(count));
        ov::Tensor ids(ov::element::i64, {1, count});
        ov::Tensor positions(ov::element::i64, {1, count});
        ov::Tensor mask(ov::element::i64, {1, past + count});
        ov::Tensor beam(ov::element::i32, {1});
        *beam.data<int32_t>() = 0;
        for (size_t i = 0; i < count; ++i) {
            ids.data<int64_t>()[i] = tokens[i];
            positions.data<int64_t>()[i] = static_cast<int64_t>(past + i);
        }
        std::fill_n(mask.data<int64_t>(), mask.get_size(), 1);
        request.set_tensor("input_ids", ids);
        request.set_tensor("position_ids", positions);
        request.set_tensor("attention_mask", mask);
        if (std::any_of(compiled.inputs().begin(), compiled.inputs().end(), [](const ov::Output<const ov::Node>& p) {
                return p.get_names().count("beam_idx");
            }))
            request.set_tensor("beam_idx", beam);
        request.infer();
        const auto result = request.get_output_tensor();
        ASSERT_EQ(result.get_element_type(), ov::element::f32);
        ASSERT_GE(result.get_size(), static_cast<size_t>(vocab));
        const auto* actual = result.data<const float>() + result.get_size() - vocab;
        const auto* expected = reference.data() + step++ * vocab;
        double error = 0, norm = 0;
        for (int32_t i = 0; i < vocab; ++i) {
            ASSERT_TRUE(std::isfinite(actual[i]));
            const double difference = actual[i] - expected[i];
            error += difference * difference;
            norm += expected[i] * expected[i];
        }
        ASSERT_GT(norm, 1e-12) << "Reference must contain nonzero logits";
        const auto predicted = std::max_element(actual, actual + vocab) - actual;
        const auto wanted = std::max_element(expected, expected + vocab) - expected;
        matching_tokens += predicted == wanted;
        RecordProperty("top1_match_step_" + std::to_string(step), predicted == wanted ? 1 : 0);
        RecordProperty("nmse_step_" + std::to_string(step), std::to_string(error / norm));
        if (override_dir) {
            // Real quantized checkpoints use lossy weight conversions. Check the first prediction
            // and continuation agreement; keep their full-logit metrics in the XML report.
            if (step == 1)
                EXPECT_EQ(predicted, wanted);
        } else {
            EXPECT_LT(error / norm, 1e-5) << "Normalized MSE against llama.cpp CPU";
        }
        past += count;
    }
    if (override_dir)
        EXPECT_GE(matching_tokens * 10, schedule.size() * 9) << "Fewer than 90% of greedy choices match";
}

INSTANTIATE_TEST_SUITE_P(Architectures,
                         GGUFArchitectureAccuracy,
                         ::testing::Values("llama",
                                           "qwen2",
                                           "qwen3",
                                           "phi3",
                                           "minicpm",
                                           "olmoe",
                                           "hunyuan-dense",
                                           "hunyuan-moe",
                                           "qwen3moe",
                                           "gemma",
                                           "gemma2",
                                           "exaone4",
                                           "ernie4_5-moe",
                                           "bailingmoe2",
                                           "maincoder",
                                           "mistral3",
                                           "smollm3",
                                           "mellum",
                                           "muse-glimmer",
                                           "deepseek2-ocr"),
                         [](const ::testing::TestParamInfo<const char*>& info) {
                             std::string name = info.param;
                             std::replace(name.begin(), name.end(), '-', '_');
                             return name;
                         });
}  // namespace
