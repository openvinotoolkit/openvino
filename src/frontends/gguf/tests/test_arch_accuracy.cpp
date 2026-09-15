// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
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
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/paged_causal_conv1d.hpp"
#include "openvino/op/paged_selective_ssm.hpp"
#include "openvino/op/selective_ssm.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"

namespace {
class GGUFArchitectureAccuracy : public ::testing::TestWithParam<const char*> {
protected:
    void compare(bool paged);
};

void GGUFArchitectureAccuracy::compare(bool paged) {
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
    const bool mamba = std::string(GetParam()).find("mamba2") == 0 || std::string(GetParam()) == "nemotron_h";
    if (mamba) {
        EXPECT_EQ(model->input("input_ids").get_partial_shape(), (ov::PartialShape{1, -1}));
        EXPECT_NO_THROW(model->input("beam_idx"));
    }
    if (paged) {
        size_t expected_scans = 0;
        for (const auto& node : model->get_ops())
            expected_scans += ov::is_type<ov::op::internal::SelectiveSSM>(node);
        ASSERT_GT(expected_scans, 0);
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::SDPAToPagedAttention>();
        ASSERT_NO_THROW(manager.run_passes(model));
        size_t scans = 0, convolutions = 0;
        for (const auto& node : model->get_ops()) {
            scans += ov::is_type<ov::op::internal::PagedSelectiveSSM>(node);
            convolutions += ov::is_type<ov::op::internal::PagedCausalConv1D>(node);
            EXPECT_NE(std::string(node->get_type_name()), "ReadValue");
            EXPECT_FALSE(ov::is_type<ov::op::internal::SelectiveSSM>(node));
            if (const auto attention = ov::as_type_ptr<ov::op::PagedAttentionExtension>(node)) {
                const auto& info = attention->get_rt_info();
                for (const size_t port : {3, 4}) {
                    const bool key = port == 3;
                    const auto heads = info.at(key ? "num_k_heads" : "num_v_heads").as<int64_t>();
                    const auto width = info.at(key ? "k_head_size" : "v_head_size").as<int64_t>();
                    const auto cache =
                        ov::as_type_ptr<ov::op::v0::Parameter>(attention->get_input_node_shared_ptr(port));
                    ASSERT_TRUE(cache);
                    cache->set_partial_shape({8, heads, 32, width});
                    cache->set_element_type(ov::element::f16);
                }
            }
        }
        EXPECT_EQ(scans, expected_scans);
        EXPECT_EQ(convolutions, expected_scans);
        for (const auto& parameter : model->get_parameters()) {
            if (parameter->get_element_type().is_dynamic())
                parameter->set_element_type(ov::element::f32);
        }
        model->validate_nodes_and_infer_types();
    }
    ov::Core core;
    auto compiled = core.compile_model(model,
                                       "CPU",
                                       ov::hint::inference_precision(ov::element::f32),
                                       ov::num_streams(1),
                                       ov::inference_num_threads(4),
                                       ov::hint::dynamic_quantization_group_size(0),
                                       ov::hint::kv_cache_precision(ov::element::f16));
    auto request = compiled.create_infer_request();
    const auto set_i32 = [&](const std::string& name, const std::vector<int32_t>& values, bool scalar = false) {
        ov::Tensor tensor(ov::element::i32, scalar ? ov::Shape{} : ov::Shape{values.size()});
        std::copy(values.begin(), values.end(), tensor.data<int32_t>());
        request.set_tensor(name, tensor);
    };
    std::vector<ov::Tensor> tables;
    if (paged) {
        for (const auto& input : compiled.inputs()) {
            const auto name = input.get_any_name();
            const bool recurrent = name.find("state_table.") != std::string::npos;
            const bool kv = name.find("key_cache.") == 0 || name.find("value_cache.") == 0;
            if (!recurrent && !kv)
                continue;
            auto shape = input.get_partial_shape();
            shape[0] = recurrent ? 2 : 8;
            ov::Tensor table(input.get_element_type(), shape.to_shape());
            std::memset(table.data(), 0, table.get_byte_size());
            request.set_tensor(name, table);
            tables.push_back(table);
        }
        set_i32("la.block_indices", {0, 0});
        set_i32("la.block_indices_begins", {0, 2});
        set_i32("la.cache_interval", {128});
        set_i32("block_indices", {0, 1, 2, 3});
        set_i32("block_indices_begins", {0, 4});
        set_i32("max_context_len", {128}, true);
    }
    size_t past = 0;
    size_t step = 0;
    size_t matching_tokens = 0;
    for (const auto& tokens : schedule) {
        const auto count = tokens.size();
        SCOPED_TRACE("past=" + std::to_string(past) + ", tokens=" + std::to_string(count));
        ov::Tensor ids(ov::element::i64, paged ? ov::Shape{count} : ov::Shape{1, count});
        ov::Tensor positions(ov::element::i64, ids.get_shape());
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
        if (paged) {
            set_i32("subsequence_begins", {0, static_cast<int32_t>(count)});
            set_i32("la.past_lens", {static_cast<int32_t>(past)});
            set_i32("past_lens", {static_cast<int32_t>(past)});
        } else {
            request.set_tensor("attention_mask", mask);
        }
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
            if (step == 1) {
                EXPECT_EQ(predicted, wanted);
            }
        } else {
            EXPECT_LT(error / norm, 1e-5) << "Normalized MSE against llama.cpp CPU";
        }
        past += count;
    }
    if (mamba && !paged) {
        const auto states = request.query_state();
        ASSERT_FALSE(states.empty());
        for (auto state : states)
            state.reset();
        const auto& tokens = schedule.front();
        ov::Tensor ids(ov::element::i64, {1, tokens.size()});
        ov::Tensor positions(ov::element::i64, ids.get_shape());
        ov::Tensor mask(ov::element::i64, ids.get_shape());
        for (size_t i = 0; i < tokens.size(); ++i) {
            ids.data<int64_t>()[i] = tokens[i];
            positions.data<int64_t>()[i] = i;
            mask.data<int64_t>()[i] = 1;
        }
        request.set_tensor("input_ids", ids);
        request.set_tensor("position_ids", positions);
        request.set_tensor("attention_mask", mask);
        request.infer();
        const auto logits = request.get_output_tensor();
        const auto* actual = logits.data<const float>() + logits.get_size() - vocab;
        double error = 0, norm = 0;
        for (int32_t i = 0; i < vocab; ++i) {
            ASSERT_TRUE(std::isfinite(actual[i]));
            error += std::pow(actual[i] - reference[i], 2);
            norm += reference[i] * reference[i];
        }
        if (override_dir)
            EXPECT_EQ(std::max_element(actual, actual + vocab) - actual,
                      std::max_element(reference.begin(), reference.begin() + vocab) - reference.begin());
        else
            EXPECT_LT(error / norm, 1e-5) << "Fresh prefill after resetting recurrent states";
    }
    if (paged && !override_dir) {
        // Two independent histories packed with unequal lengths. Reverse their physical
        // state rows and reuse the same request after the preceding single-sequence run.
        for (auto& table : tables)
            std::memset(table.data(), 0, table.get_byte_size());
        set_i32("la.block_indices", {1, 1, 0, 0});
        set_i32("la.block_indices_begins", {0, 2, 4});
        set_i32("la.cache_interval", {128, 128});
        set_i32("block_indices", {0, 1, 2, 3, 4, 5, 6, 7});
        set_i32("block_indices_begins", {0, 4, 8});
        for (const bool decode : {false, true}) {
            const std::vector<int64_t> tokens =
                decode ? std::vector<int64_t>{4, 5, 6} : std::vector<int64_t>{1, 2, 3, 1, 2, 3, 4};
            const size_t first_count = decode ? 1 : 3;
            ov::Tensor ids(ov::element::i64, {tokens.size()});
            std::copy(tokens.begin(), tokens.end(), ids.data<int64_t>());
            request.set_tensor("input_ids", ids);
            ov::Tensor positions(ov::element::i64, {tokens.size()});
            for (size_t i = 0; i < tokens.size(); ++i)
                positions.data<int64_t>()[i] =
                    i < first_count ? i + (decode ? 3 : 0) : i - first_count + (decode ? 4 : 0);
            request.set_tensor("position_ids", positions);
            set_i32("subsequence_begins", {0, static_cast<int32_t>(first_count), static_cast<int32_t>(tokens.size())});
            set_i32("la.past_lens", decode ? std::vector<int32_t>{3, 4} : std::vector<int32_t>{0, 0});
            set_i32("past_lens", decode ? std::vector<int32_t>{3, 4} : std::vector<int32_t>{0, 0});
            request.infer();
            const auto logits = request.get_output_tensor();
            ASSERT_EQ(logits.get_shape(), (ov::Shape{1, tokens.size(), static_cast<size_t>(vocab)}));
            for (size_t sequence = 0; sequence < 2; ++sequence) {
                const size_t last_token = sequence == 0 ? first_count - 1 : tokens.size() - 1;
                const auto* actual = logits.data<const float>() + last_token * vocab;
                const auto* expected = reference.data() + (sequence + (decode ? 1 : 0)) * vocab;
                double error = 0, norm = 0;
                for (int32_t i = 0; i < vocab; ++i) {
                    ASSERT_TRUE(std::isfinite(actual[i]));
                    error += std::pow(actual[i] - expected[i], 2);
                    norm += expected[i] * expected[i];
                }
                EXPECT_LT(error / norm, 1e-5) << "Packed sequence " << sequence << ", decode=" << decode;
            }
        }
    }

    if (override_dir) {
        EXPECT_GE(matching_tokens * 10, schedule.size() * 9) << "Fewer than 90% of greedy choices match";
    }
}

TEST_P(GGUFArchitectureAccuracy, PrefillAndCachedDecodeMatchLlamaCPU) {
    compare(false);
}

class GGUFMambaPagedAccuracy : public GGUFArchitectureAccuracy {};
TEST_P(GGUFMambaPagedAccuracy, PrefillAndCachedDecodeMatchLlamaCPU) {
    compare(true);
}
INSTANTIATE_TEST_SUITE_P(Architectures,
                         GGUFMambaPagedAccuracy,
                         ::testing::Values("mamba2", "mamba2-tied", "nemotron_h"),
                         [](const ::testing::TestParamInfo<const char*>& info) {
                             std::string name = info.param;
                             std::replace(name.begin(), name.end(), '-', '_');
                             return name;
                         });

INSTANTIATE_TEST_SUITE_P(Architectures,
                         GGUFArchitectureAccuracy,
                         ::testing::Values("nemotron_h",
                                           "mamba2",
                                           "mamba2-tied",
                                           "llama",
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
                                           "deepseek2-ocr",
                                           "devstral-small",
                                           "devstral-small2",
                                           "devstral2"),
                         [](const ::testing::TestParamInfo<const char*>& info) {
                             std::string name = info.param;
                             std::replace(name.begin(), name.end(), '-', '_');
                             return name;
                         });
}  // namespace
