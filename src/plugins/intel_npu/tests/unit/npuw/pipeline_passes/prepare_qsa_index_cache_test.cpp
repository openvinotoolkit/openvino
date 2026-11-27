// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/prepare_qsa_index_cache.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <unordered_map>

#include "llm_pass_test_fixture.hpp"
#include "npuw_transformations/lower_qsa_indexing.hpp"
#include "npuw_transformations/lower_sparse_attention.hpp"
#include "npuw_transformations/mask_linear_attention_padding.hpp"
#include "npuw_transformations/qwen_sparse_attention_fusion.hpp"
#include "npuw_transformations/reshape_to_static.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/qsa_prepare.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/stateful_to_stateless.hpp"

namespace {

using namespace ov;
namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v4 = ov::op::v4;
namespace v8 = ov::op::v8;
namespace v12 = ov::op::v12;

std::shared_ptr<Model> make_cache_model(bool supported) {
    auto blocks = std::make_shared<v0::Parameter>(element::f32, PartialShape{1, -1, 4});
    blocks->output(0).set_names({"cache_params.past.block_index_key.0"});
    auto pending = std::make_shared<v0::Parameter>(element::f32, PartialShape{1, 3, 4});
    pending->output(0).set_names({"cache_params.past.incomplete_index_key.0"});
    auto c = [](int64_t value) {
        return v0::Constant::create(element::i64, Shape{}, {value});
    };
    auto cv = [](int64_t value) {
        return v0::Constant::create(element::i64, Shape{1}, {value});
    };
    auto result = std::make_shared<v0::Result>(blocks);
    if (supported) {
        auto total = c(3);
        auto end = std::make_shared<v1::Add>(total, c(1));
        auto positions = std::make_shared<v4::Range>(c(0), end, c(1), element::i64);
        auto row = std::make_shared<v0::Unsqueeze>(positions, cv(0));
        auto old_count = v0::Constant::create(element::i64, Shape{1, 1}, {2});
        auto mask = std::make_shared<v0::Unsqueeze>(std::make_shared<v1::Less>(row, old_count), cv(-1));
        auto zeros = v0::Constant::create(element::f32, Shape{1, 4, 4}, {0});
        auto concat = std::make_shared<v0::Concat>(OutputVector{blocks, zeros}, 1);
        auto end_vector = std::make_shared<v0::Unsqueeze>(end, cv(0));
        auto input_slice = std::make_shared<v8::Slice>(concat, cv(0), end_vector, cv(1), cv(1));
        auto buffer = std::make_shared<v1::Select>(mask, input_slice, v0::Constant::create(element::f32, Shape{}, {0}));
        auto valid = v0::Constant::create(element::boolean, Shape{1, 1}, {true});
        auto index = std::make_shared<v1::Select>(valid, old_count, total);
        auto index_expanded = std::make_shared<v0::Unsqueeze>(index, cv(-1));
        auto indices =
            std::make_shared<v3::Broadcast>(index_expanded, v0::Constant::create(element::i64, Shape{3}, {1, 1, 4}));
        auto updates = v0::Constant::create(element::f32, Shape{1, 1, 4}, {9});
        auto scatter = std::make_shared<v12::ScatterElementsUpdate>(buffer, indices, updates, c(1));
        auto final_end = std::make_shared<v0::Unsqueeze>(total, cv(0));
        auto output_slice = std::make_shared<v8::Slice>(scatter, cv(0), final_end, cv(1), cv(1));
        result = std::make_shared<v0::Result>(output_slice);
    }
    result->output(0).set_names({"cache_params.present.block_index_key.0"});
    return std::make_shared<Model>(ResultVector{result}, ParameterVector{blocks, pending});
}

size_t exported_sparse_count(const std::shared_ptr<Model>& model) {
    size_t count = 0;
    for (const auto& node : model->get_ordered_ops()) {
        count += ov::is_type<ov::op::v17::SparseAttention>(node);
    }
    return count;
}

bool is_sparse_score(const std::shared_ptr<Node>& node) {
    const std::string suffix = "/sparse_scores";
    const auto& name = node->get_friendly_name();
    return ov::is_type<v0::MatMul>(node) && name.size() >= suffix.size() &&
           name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void lower_real_qsa(const std::shared_ptr<Model>& model, bool dense = false) {
    const auto exported = exported_sparse_count(model);
    ov::npuw::QwenSparseAttentionFusion fusion;
    fusion.run_on_model(model);
    ASSERT_EQ(exported + fusion.fused_count(), 8u);
    if (exported) {
        ASSERT_TRUE(ov::npuw::LowerQSAPrepare().run_on_model(model));
        model->validate_nodes_and_infer_types();
    }
    const auto mask_parameter = model->input("attention_mask").get_node_shared_ptr();
    if (dense) {
        ASSERT_TRUE(ov::npuw::LowerSparseAttention(mask_parameter).run_on_model(model));
    } else {
        ASSERT_TRUE(ov::npuw::LowerSparseAttentionToGathered(mask_parameter).run_on_model(model));
    }
    fusion.lower_indexers();
    if (exported) {
        ASSERT_TRUE(ov::npuw::LowerSparseAttentionIndexer().run_on_model(model));
    }
    model->validate_nodes_and_infer_types();
}

TEST(PrepareQSAIndexCacheTest, RealQwen4PaddedPrefillMatchesUnpaddedReference) {
    const auto* path = std::getenv("NPUW_QSA_TEST_MODEL");
    if (!path) {
        GTEST_SKIP() << "Set NPUW_QSA_TEST_MODEL to the Qwen4 language-model XML";
    }
    Core core;
    auto model = core.read_model(path);
    auto reference_model = model->clone();
    constexpr size_t prompt = 4;
    constexpr size_t hidden = 2560;
    // Compare padding with the same gathered implementation; other attention kernels can round differently.
    ASSERT_TRUE(ov::pass::StatefulToStateless().run_on_model(reference_model));
    ASSERT_TRUE(ov::npuw::PrepareQSAIndexCache(prompt).run_on_model(reference_model));
    ASSERT_TRUE(
        ov::npuw::ReshapeToStatic(prompt, prompt, ov::npuw::KVAxesPosition{0u, 2u}, 0u).run_on_model(reference_model));
    ASSERT_NO_FATAL_FAILURE(lower_real_qsa(reference_model));
    ASSERT_TRUE(ov::npuw::MaskLinearAttentionPadding().run_on_model(reference_model));
    reference_model->validate_nodes_and_infer_types();
    ASSERT_TRUE(ov::pass::StatefulToStateless().run_on_model(model));
    ASSERT_TRUE(ov::npuw::PrepareQSAIndexCache(64).run_on_model(model));
    ASSERT_TRUE(ov::npuw::ReshapeToStatic(64, 64, ov::npuw::KVAxesPosition{0u, 2u}, 0u).run_on_model(model));
    ASSERT_NO_FATAL_FAILURE(lower_real_qsa(model));
    ASSERT_TRUE(ov::npuw::MaskLinearAttentionPadding().run_on_model(model));
    model->validate_nodes_and_infer_types();

    struct InferenceData {
        std::vector<float> logits;
        std::unordered_map<std::string, std::vector<float>> recurrent_states;
        std::unordered_map<std::string, Tensor> kv_states;
    };
    auto is_recurrent_state = [](const std::string& name) {
        return name.find(".block_index_key.") != std::string::npos ||
               name.find(".incomplete_index_key.") != std::string::npos ||
               name.find(".incomplete_index_rope.") != std::string::npos || name.find(".conv.") != std::string::npos ||
               name.find(".ssm.") != std::string::npos;
    };
    auto is_kv_state = [](const std::string& name) {
        return name.find(".key.") != std::string::npos || name.find(".value.") != std::string::npos ||
               (name.size() >= 4 && name.compare(name.size() - 4, 4, ".key") == 0) ||
               (name.size() >= 6 && name.compare(name.size() - 6, 6, ".value") == 0);
    };
    auto run = [&](const std::shared_ptr<Model>& target, size_t width) {
        auto compiled = core.compile_model(target, "CPU", hint::inference_precision(element::f32));
        auto request = compiled.create_infer_request();
        for (const auto& port : compiled.inputs()) {
            const auto names = port.get_names();
            Tensor tensor(port.get_element_type(), port.get_shape());
            if (tensor.get_byte_size() != 0) {
                std::memset(tensor.data(), 0, tensor.get_byte_size());
            }
            if (names.count("attention_mask")) {
                std::fill_n(tensor.data<int64_t>() + width - prompt, prompt, 1);
            } else if (names.count("position_ids")) {
                for (size_t plane = 0; plane < 4; ++plane) {
                    for (size_t token = 0; token < prompt; ++token) {
                        tensor.data<int64_t>()[plane * width + width - prompt + token] = static_cast<int64_t>(token);
                    }
                }
            } else if (names.count("inputs_embeds")) {
                auto* values = tensor.data<float>() + (width - prompt) * hidden;
                for (size_t i = 0; i < prompt * hidden; ++i) {
                    values[i] = static_cast<float>(static_cast<int>(i % 31) - 15) * 0.001f;
                }
            }
            request.set_tensor(port, tensor);
        }
        request.infer();
        const auto output = request.get_tensor("logits");
        const auto* last = output.data<const float>() + output.get_size() - output.get_shape().back();
        InferenceData data;
        data.logits.assign(last, last + output.get_shape().back());
        for (const auto& port : compiled.outputs()) {
            for (const auto& name : port.get_names()) {
                if (!is_recurrent_state(name) && !is_kv_state(name)) {
                    continue;
                }
                const auto tensor = request.get_tensor(port);
                OPENVINO_ASSERT(tensor.get_element_type() == element::f32);
                if (is_kv_state(name)) {
                    data.kv_states.emplace(name, tensor);
                } else {
                    const auto* values = tensor.data<const float>();
                    data.recurrent_states.emplace(name, std::vector<float>(values, values + tensor.get_size()));
                }
            }
        }
        return data;
    };

    const auto reference_data = run(reference_model, prompt);
    const auto actual_data = run(model, 64);
    const auto& reference = reference_data.logits;
    const auto& actual = actual_data.logits;
    ASSERT_EQ(actual.size(), reference.size());
    double dot = 0.0, reference_norm = 0.0, actual_norm = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        dot += static_cast<double>(reference[i]) * actual[i];
        reference_norm += static_cast<double>(reference[i]) * reference[i];
        actual_norm += static_cast<double>(actual[i]) * actual[i];
    }
    ASSERT_GT(reference_norm, 0.0);
    ASSERT_GT(actual_norm, 0.0);
    EXPECT_GT(dot / std::sqrt(reference_norm * actual_norm), 0.9999);
    EXPECT_EQ(std::distance(reference.begin(), std::max_element(reference.begin(), reference.end())),
              std::distance(actual.begin(), std::max_element(actual.begin(), actual.end())));
    ASSERT_EQ(reference_data.recurrent_states.size(), 72u);
    ASSERT_EQ(actual_data.recurrent_states.size(), reference_data.recurrent_states.size());
    for (const auto& state : reference_data.recurrent_states) {
        SCOPED_TRACE(state.first);
        const auto it = actual_data.recurrent_states.find(state.first);
        ASSERT_NE(it, actual_data.recurrent_states.end());
        ASSERT_LE(state.second.size(), it->second.size());
        float max_error = 0.f;
        for (size_t i = 0; i < state.second.size(); ++i) {
            max_error = std::max(max_error, std::abs(state.second[i] - it->second[i]));
        }
        EXPECT_LT(max_error, 1e-4f);
    }
    ASSERT_EQ(reference_data.kv_states.size(), 16u);
    ASSERT_EQ(actual_data.kv_states.size(), reference_data.kv_states.size());
    for (const auto& state : reference_data.kv_states) {
        SCOPED_TRACE(state.first);
        const auto it = actual_data.kv_states.find(state.first);
        ASSERT_NE(it, actual_data.kv_states.end());
        const auto& expected_shape = state.second.get_shape();
        const auto& actual_shape = it->second.get_shape();
        ASSERT_EQ(expected_shape.size(), 4u);
        ASSERT_EQ(actual_shape.size(), 4u);
        ASSERT_EQ(expected_shape[0], 1u);
        ASSERT_EQ(actual_shape[0], 1u);
        ASSERT_EQ(expected_shape[1], actual_shape[1]);
        ASSERT_EQ(expected_shape[3], actual_shape[3]);
        ASSERT_LE(expected_shape[2], actual_shape[2]);
        const auto* expected_values = state.second.data<const float>();
        const auto* actual_values = it->second.data<const float>();
        float max_error = 0.f;
        for (size_t head = 0; head < expected_shape[1]; ++head) {
            for (size_t i = 0; i < expected_shape[2] * expected_shape[3]; ++i) {
                const auto expected_index = head * expected_shape[2] * expected_shape[3] + i;
                const auto actual_index =
                    (head * actual_shape[2] + actual_shape[2] - expected_shape[2]) * actual_shape[3] + i;
                max_error =
                    std::max(max_error, std::abs(expected_values[expected_index] - actual_values[actual_index]));
            }
        }
        EXPECT_LT(max_error, 1e-4f);
    }
}

TEST(PrepareQSAIndexCacheTest, RealQwen4StaticDecodeMatchesStatefulReference) {
    const auto* path = std::getenv("NPUW_QSA_TEST_MODEL");
    if (!path) {
        GTEST_SKIP() << "Set NPUW_QSA_TEST_MODEL to the Qwen4 language-model XML";
    }
    constexpr size_t prompt = 4;
    constexpr size_t hidden = 2560;
    constexpr size_t capacity = 128;
    Core core;
    auto model = core.read_model(path);
    auto reference_model = model->clone();
    if (exported_sparse_count(reference_model)) {
        const auto supported = core.query_model(reference_model, "CPU");
        for (const auto& node : reference_model->get_ordered_ops()) {
            if (ov::is_type<ov::op::v17::QSAPrepare>(node) && supported.count(node->get_friendly_name()) == 0) {
                GTEST_SKIP() << "The CPU plugin does not support the exported QSAPrepare reference; "
                                "RealQwen4GatheredDecodeMatchesDenseReference tests the NPUW paths independently.";
            }
        }
    }
    ASSERT_TRUE(ov::pass::StatefulToStateless().run_on_model(model));
    ASSERT_TRUE(ov::npuw::PrepareQSAIndexCache(capacity).run_on_model(model));
    ASSERT_TRUE(ov::npuw::ReshapeToStatic(1, capacity, ov::npuw::KVAxesPosition{0u, 2u}, 0u).run_on_model(model));
    ASSERT_NO_FATAL_FAILURE(lower_real_qsa(model));

    auto reference_compiled = core.compile_model(reference_model,
                                                 "CPU",
                                                 hint::inference_precision(element::f32),
                                                 hint::kv_cache_precision(element::f32));
    auto compiled = core.compile_model(model, "CPU", hint::inference_precision(element::f32));
    auto reference = reference_compiled.create_infer_request();
    auto actual = compiled.create_infer_request();
    Tensor embeddings(element::f32, Shape{1, prompt, hidden});
    for (size_t i = 0; i < embeddings.get_size(); ++i) {
        embeddings.data<float>()[i] = static_cast<float>(static_cast<int>(i % 31) - 15) * 0.001f;
    }
    Tensor mask(element::i64, Shape{1, prompt});
    std::fill_n(mask.data<int64_t>(), prompt, 1);
    Tensor positions(element::i64, Shape{4, 1, prompt});
    for (size_t plane = 0; plane < 4; ++plane) {
        for (size_t token = 0; token < prompt; ++token) {
            positions.data<int64_t>()[plane * prompt + token] = static_cast<int64_t>(token);
        }
    }
    Tensor beam(element::i32, Shape{1});
    beam.data<int32_t>()[0] = 0;
    reference.set_tensor("inputs_embeds", embeddings);
    reference.set_tensor("attention_mask", mask);
    reference.set_tensor("position_ids", positions);
    reference.set_tensor("beam_idx", beam);
    reference.infer();

    for (const auto& port : compiled.inputs()) {
        auto tensor = actual.get_tensor(port);
        if (tensor.get_byte_size()) {
            std::memset(tensor.data(), 0, tensor.get_byte_size());
        }
    }
    size_t copied_states = 0;
    for (const auto& state : reference.query_state()) {
        const auto id = state.get_name();
        const std::string past_prefix = "cache_params.past.";
        const auto present_pos = id.find("cache_params.present.");
        ASSERT_EQ(id.find(past_prefix), 0u);
        ASSERT_NE(present_pos, std::string::npos);
        const auto suffix = id.substr(past_prefix.size(), present_pos - past_prefix.size());
        const bool kv = suffix.find("key.") == 0 || suffix.find("value.") == 0;
        std::string input_name = past_prefix + suffix;
        if (kv) {
            const auto dot = suffix.find('.');
            input_name = "past_key_values." + suffix.substr(dot + 1) + "." + suffix.substr(0, dot);
        }
        auto destination = actual.get_tensor(input_name);
        const auto source = state.get_state();
        ASSERT_EQ(source.get_element_type(), element::f32);
        ASSERT_EQ(destination.get_element_type(), element::f32);
        if (kv) {
            const auto& src_shape = source.get_shape();
            const auto& dst_shape = destination.get_shape();
            ASSERT_EQ(src_shape.size(), 4u);
            ASSERT_EQ(dst_shape.size(), 4u);
            ASSERT_EQ(src_shape[2], prompt);
            ASSERT_EQ(dst_shape[2], capacity - 1);
            ASSERT_EQ(src_shape[1], dst_shape[1]);
            ASSERT_EQ(src_shape[3], dst_shape[3]);
            for (size_t head = 0; head < src_shape[1]; ++head) {
                std::copy_n(source.data<const float>() + head * prompt * src_shape[3],
                            prompt * src_shape[3],
                            destination.data<float>() + head * dst_shape[2] * dst_shape[3]);
            }
        } else {
            ASSERT_LE(source.get_byte_size(), destination.get_byte_size());
            std::memcpy(destination.data(), source.data(), source.get_byte_size());
        }
        ++copied_states;
    }
    ASSERT_EQ(copied_states, 88u);

    for (size_t step = 0; step < 5; ++step) {
        SCOPED_TRACE(step);
        const auto position = prompt + step;
        Tensor next_embedding(element::f32, Shape{1, 1, hidden});
        for (size_t i = 0; i < hidden; ++i) {
            next_embedding.data<float>()[i] = static_cast<float>(static_cast<int>((i + step * 7) % 29) - 14) * 0.001f;
        }
        Tensor next_positions(element::i64, Shape{4, 1, 1});
        std::fill_n(next_positions.data<int64_t>(), 4, static_cast<int64_t>(position));
        Tensor reference_mask(element::i64, Shape{1, position + 1});
        std::fill_n(reference_mask.data<int64_t>(), position + 1, 1);
        reference.set_tensor("inputs_embeds", next_embedding);
        reference.set_tensor("position_ids", next_positions);
        reference.set_tensor("attention_mask", reference_mask);
        reference.infer();

        auto physical_mask = actual.get_tensor("attention_mask");
        std::fill_n(physical_mask.data<int64_t>(), capacity, 0);
        std::fill_n(physical_mask.data<int64_t>(), position, 1);
        physical_mask.data<int64_t>()[capacity - 1] = 1;
        actual.set_tensor("inputs_embeds", next_embedding);
        actual.set_tensor("position_ids", next_positions);
        actual.infer();
        const auto expected = reference.get_tensor("logits");
        const auto observed = actual.get_tensor("logits");
        ASSERT_EQ(expected.get_size(), observed.get_size());
        double dot = 0., expected_norm = 0., observed_norm = 0.;
        for (size_t i = 0; i < expected.get_size(); ++i) {
            const double x = expected.data<const float>()[i];
            const double y = observed.data<const float>()[i];
            dot += x * y;
            expected_norm += x * x;
            observed_norm += y * y;
        }
        ASSERT_GT(expected_norm, 0.);
        ASSERT_GT(observed_norm, 0.);
        // Static and stateful CPU execution use different kernel shapes; this model amplifies FP32 rounding.
        EXPECT_GT(dot / std::sqrt(expected_norm * observed_norm), 0.997);
        const auto* expected_values = expected.data<const float>();
        const auto* observed_values = observed.data<const float>();
        EXPECT_EQ(std::max_element(expected_values, expected_values + expected.get_size()) - expected_values,
                  std::max_element(observed_values, observed_values + observed.get_size()) - observed_values);

        for (const auto& port : compiled.inputs()) {
            const auto name = port.get_any_name();
            if (name.find("cache_params.past.") == 0) {
                auto input = actual.get_tensor(port);
                const auto output_name =
                    "cache_params.present." + name.substr(std::string("cache_params.past.").size());
                const auto output = actual.get_tensor(output_name);
                ASSERT_EQ(input.get_byte_size(), output.get_byte_size());
                std::memcpy(input.data(), output.data(), input.get_byte_size());
            } else if (name.find("past_key_values.") == 0) {
                auto input = actual.get_tensor(port);
                const auto output_name = "present." + name.substr(std::string("past_key_values.").size());
                const auto output = actual.get_tensor(output_name);
                const auto& shape = input.get_shape();
                ASSERT_EQ(shape.size(), 4u);
                for (size_t head = 0; head < shape[1]; ++head) {
                    std::copy_n(output.data<const float>() + (head * capacity + capacity - 1) * shape[3],
                                shape[3],
                                input.data<float>() + (head * shape[2] + position) * shape[3]);
                }
            }
        }
    }
}

TEST(PrepareQSAIndexCacheTest, RealQwen4GatheredDecodeMatchesDenseReference) {
    const auto* path = std::getenv("NPUW_QSA_TEST_MODEL");
    if (!path) {
        GTEST_SKIP() << "Set NPUW_QSA_TEST_MODEL to the Qwen4 language-model XML";
    }
    constexpr size_t capacity = 128;
    constexpr size_t hidden = 2560;
    Core core;
    auto gathered_model = core.read_model(path);
    auto dense_model = gathered_model->clone();
    for (const auto& model : {gathered_model, dense_model}) {
        ASSERT_TRUE(ov::pass::StatefulToStateless().run_on_model(model));
        ASSERT_TRUE(ov::npuw::PrepareQSAIndexCache(capacity).run_on_model(model));
        ASSERT_TRUE(ov::npuw::ReshapeToStatic(1, capacity, ov::npuw::KVAxesPosition{0u, 2u}, 0u).run_on_model(model));
    }
    ASSERT_NO_FATAL_FAILURE(lower_real_qsa(gathered_model));
    ASSERT_NO_FATAL_FAILURE(lower_real_qsa(dense_model, true));

    auto gathered_compiled = core.compile_model(gathered_model, "CPU", hint::inference_precision(element::f32));
    auto dense_compiled = core.compile_model(dense_model, "CPU", hint::inference_precision(element::f32));
    const auto embedding_path = std::filesystem::path(path).parent_path() / "openvino_text_embeddings_model.xml";
    ASSERT_TRUE(std::filesystem::exists(embedding_path)) << embedding_path.string();
    auto embedding_model = core.read_model(embedding_path.string());
    auto embedding_compiled = core.compile_model(embedding_model, "CPU", hint::inference_precision(element::f32));
    auto embedding_request = embedding_compiled.create_infer_request();
    auto gathered = gathered_compiled.create_infer_request();
    auto dense = dense_compiled.create_infer_request();
    for (auto* request : {&gathered, &dense}) {
        const auto compiled = request->get_compiled_model();
        for (const auto& port : compiled.inputs()) {
            auto tensor = request->get_tensor(port);
            if (tensor.get_byte_size()) {
                std::memset(tensor.data(), 0, tensor.get_byte_size());
            }
        }
    }

    const std::vector<int64_t> token_ids{760, 6511, 314, 9338, 369, 11751};
    for (size_t step = 0; step < token_ids.size(); ++step) {
        SCOPED_TRACE(step);
        Tensor token(element::i64, Shape{1, 1});
        token.data<int64_t>()[0] = token_ids[step];
        embedding_request.set_input_tensor(token);
        embedding_request.infer();
        const auto embedding = embedding_request.get_output_tensor();
        ASSERT_EQ(embedding.get_shape(), (Shape{1, 1, hidden}));
        Tensor positions(element::i64, Shape{4, 1, 1});
        std::fill_n(positions.data<int64_t>(), 4, static_cast<int64_t>(step));
        for (auto* request : {&gathered, &dense}) {
            auto mask = request->get_tensor("attention_mask");
            std::fill_n(mask.data<int64_t>(), capacity, 0);
            std::fill_n(mask.data<int64_t>(), step, 1);
            mask.data<int64_t>()[capacity - 1] = 1;
            request->set_tensor("inputs_embeds", embedding);
            request->set_tensor("position_ids", positions);
            request->infer();
        }
        const auto expected = dense.get_tensor("logits");
        const auto actual = gathered.get_tensor("logits");
        ASSERT_EQ(expected.get_shape(), actual.get_shape());
        double dot = 0., expected_norm = 0., actual_norm = 0.;
        for (size_t i = 0; i < expected.get_size(); ++i) {
            const double x = expected.data<const float>()[i];
            const double y = actual.data<const float>()[i];
            dot += x * y;
            expected_norm += x * x;
            actual_norm += y * y;
        }
        ASSERT_GT(expected_norm, 0.);
        ASSERT_GT(actual_norm, 0.);
        EXPECT_GT(dot / std::sqrt(expected_norm * actual_norm), 0.997);
        EXPECT_EQ(std::max_element(expected.data<const float>(), expected.data<const float>() + expected.get_size()) -
                      expected.data<const float>(),
                  std::max_element(actual.data<const float>(), actual.data<const float>() + actual.get_size()) -
                      actual.data<const float>());

        // Use the same reference cache on the next step so accumulated FP32 rounding does not obscure attention errors.
        for (auto* request : {&gathered, &dense}) {
            const auto compiled = request->get_compiled_model();
            for (const auto& port : compiled.inputs()) {
                const auto name = port.get_any_name();
                if (name.find("cache_params.past.") == 0) {
                    auto input = request->get_tensor(port);
                    const auto output_name =
                        "cache_params.present." + name.substr(std::string("cache_params.past.").size());
                    const auto output = dense.get_tensor(output_name);
                    ASSERT_EQ(input.get_byte_size(), output.get_byte_size());
                    std::memcpy(input.data(), output.data(), input.get_byte_size());
                } else if (name.find("past_key_values.") == 0) {
                    auto input = request->get_tensor(port);
                    const auto output_name = "present." + name.substr(std::string("past_key_values.").size());
                    const auto output = dense.get_tensor(output_name);
                    const auto& shape = input.get_shape();
                    ASSERT_EQ(shape.size(), 4u);
                    for (size_t head = 0; head < shape[1]; ++head) {
                        std::copy_n(output.data<const float>() + (head * capacity + capacity - 1) * shape[3],
                                    shape[3],
                                    input.data<float>() + (head * shape[2] + step) * shape[3]);
                    }
                }
            }
        }
    }
}

TEST(PrepareQSAIndexCacheTest, PreservesValidBlocksAndMakesCapacityStatic) {
    auto model = make_cache_model(true);
    ASSERT_TRUE(ov::npuw::PrepareQSAIndexCache(16).run_on_model(model));
    model->reshape({{"cache_params.past.block_index_key.0", PartialShape{1, 4, 4}}});
    EXPECT_EQ(model->output().get_shape(), (Shape{1, 4, 4}));
    Core core;
    auto compiled = core.compile_model(model, "CPU", hint::inference_precision(element::f32));
    auto request = compiled.create_infer_request();
    Tensor blocks(element::f32, Shape{1, 4, 4});
    for (size_t block = 0; block < 4; ++block) {
        std::fill_n(blocks.data<float>() + block * 4, 4, static_cast<float>(block + 1));
    }
    Tensor pending(element::f32, Shape{1, 3, 4});
    std::fill_n(pending.data<float>(), pending.get_size(), 0.0f);
    request.set_tensor("cache_params.past.block_index_key.0", blocks);
    request.set_tensor("cache_params.past.incomplete_index_key.0", pending);
    request.infer();
    const auto output = request.get_output_tensor();
    for (size_t block = 0; block < 4; ++block) {
        const float expected = block < 2 ? static_cast<float>(block + 1) : block == 2 ? 9.0f : 0.0f;
        for (size_t i = 0; i < 4; ++i) {
            EXPECT_FLOAT_EQ(output.data<const float>()[block * 4 + i], expected);
        }
    }
}

TEST(PrepareQSAIndexCacheTest, RejectsUnrecognizedStateGraphWithoutChangingIt) {
    auto model = make_cache_model(false);
    EXPECT_THROW(ov::npuw::PrepareQSAIndexCache(16).run_on_model(model), ov::Exception);
    EXPECT_TRUE(ov::is_type<v0::Parameter>(model->get_results()[0]->get_input_node_shared_ptr(0)));
}

TEST(PrepareQSAIndexCacheTest, RealQwen4ModelHasStaticBlockCaches) {
    const auto* path = std::getenv("NPUW_QSA_TEST_MODEL");
    if (!path) {
        GTEST_SKIP() << "Set NPUW_QSA_TEST_MODEL to the Qwen4 language-model XML";
    }
    Core core;
    auto model = core.read_model(path);
    ASSERT_TRUE(ov::pass::StatefulToStateless().run_on_model(model));
    ASSERT_TRUE(ov::npuw::PrepareQSAIndexCache(64).run_on_model(model));
    ASSERT_TRUE(ov::npuw::ReshapeToStatic(64, 64, ov::npuw::KVAxesPosition{0u, 2u}, 0u).run_on_model(model));
    if (exported_sparse_count(model)) {
        ASSERT_TRUE(ov::npuw::LowerQSAPrepare().run_on_model(model));
        model->validate_nodes_and_infer_types();
    }
    for (size_t layer = 0; layer < 8; ++layer) {
        const auto name = "cache_params.past.block_index_key." + std::to_string(layer);
        EXPECT_EQ(model->input(name).get_shape(), (Shape{1, 16, 128}));
        EXPECT_EQ(model->output("cache_params.present.block_index_key." + std::to_string(layer)).get_shape(),
                  (Shape{1, 16, 128}));
    }
}

TEST(PrepareQSAIndexCacheTest, RealQwen4ModelFusesAndLowersSparseAttention) {
    const auto* path = std::getenv("NPUW_QSA_TEST_MODEL");
    if (!path) {
        GTEST_SKIP() << "Set NPUW_QSA_TEST_MODEL to the Qwen4 language-model XML";
    }
    Core core;
    auto model = core.read_model(path);
    ASSERT_TRUE(ov::pass::StatefulToStateless().run_on_model(model));
    ASSERT_TRUE(ov::npuw::PrepareQSAIndexCache(64).run_on_model(model));
    ASSERT_TRUE(ov::npuw::ReshapeToStatic(64, 64, ov::npuw::KVAxesPosition{0u, 2u}, 0u).run_on_model(model));
    ASSERT_NO_FATAL_FAILURE(lower_real_qsa(model));
    size_t score_count = 0;
    for (const auto& node : model->get_ordered_ops()) {
        EXPECT_FALSE(ov::is_type<ov::op::v17::SparseAttention>(node));
        EXPECT_FALSE(ov::is_type<ov::op::v17::SparseAttentionIndexer>(node));
        EXPECT_FALSE(ov::is_type<ov::op::v17::QSAPrepare>(node));
        EXPECT_FALSE(ov::is_type<ov::op::v13::ScaledDotProductAttention>(node));
        if (is_sparse_score(node)) {
            ++score_count;
            EXPECT_EQ(node->get_output_partial_shape(0).rank().get_length(), 5);
        }
    }
    EXPECT_EQ(score_count, 8);
}

TEST(PrepareQSAIndexCacheTest, RealQwen4RejectsUnstableIndexerTopK) {
    const auto* path = std::getenv("NPUW_QSA_TEST_MODEL");
    if (!path) {
        GTEST_SKIP() << "Set NPUW_QSA_TEST_MODEL to the Qwen4 language-model XML";
    }
    Core core;
    auto model = core.read_model(path);
    if (exported_sparse_count(model)) {
        GTEST_SKIP() << "The exported fused graph does not contain the legacy TopK pattern";
    }
    ASSERT_TRUE(ov::pass::StatefulToStateless().run_on_model(model));
    ASSERT_TRUE(ov::npuw::PrepareQSAIndexCache(64).run_on_model(model));
    ASSERT_TRUE(ov::npuw::ReshapeToStatic(64, 64, ov::npuw::KVAxesPosition{0u, 2u}, 0u).run_on_model(model));
    bool changed = false;
    for (const auto& node : model->get_ordered_ops()) {
        if (const auto topk = ov::as_type_ptr<ov::op::v11::TopK>(node)) {
            topk->set_stable(false);
            changed = true;
            break;
        }
    }
    ASSERT_TRUE(changed);
    ov::npuw::QwenSparseAttentionFusion fusion;
    ASSERT_TRUE(fusion.run_on_model(model));
    EXPECT_EQ(fusion.fused_count(), 7);
}

TEST(PrepareQSAIndexCacheTest, RealQwen4LoweredMasksMatchOriginalMasks) {
    const auto* path = std::getenv("NPUW_QSA_TEST_MODEL");
    if (!path) {
        GTEST_SKIP() << "Set NPUW_QSA_TEST_MODEL to the Qwen4 language-model XML";
    }
    Core core;
    auto model = core.read_model(path);
    if (exported_sparse_count(model)) {
        GTEST_SKIP() << "The exported fused graph does not contain the legacy SDPA mask";
    }
    ASSERT_TRUE(ov::pass::StatefulToStateless().run_on_model(model));
    ASSERT_TRUE(ov::npuw::PrepareQSAIndexCache(64).run_on_model(model));
    ASSERT_TRUE(ov::npuw::ReshapeToStatic(64, 64, ov::npuw::KVAxesPosition{0u, 2u}, 0u).run_on_model(model));
    std::unordered_map<std::string, Output<Node>> original_masks;
    for (const auto& node : model->get_ordered_ops()) {
        if (const auto attention = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(node)) {
            original_masks.emplace(attention->get_friendly_name(), attention->input_value(3));
        }
    }
    ov::npuw::QwenSparseAttentionFusion fusion;
    ASSERT_TRUE(fusion.run_on_model(model));
    ASSERT_EQ(fusion.fused_count(), 8);
    auto physical_mask = model->input("attention_mask").get_node_shared_ptr();
    ASSERT_TRUE(ov::npuw::LowerSparseAttention(physical_mask).run_on_model(model));
    fusion.lower_indexers();

    ResultVector extra_results;
    size_t index = 0;
    for (const auto& node : model->get_ordered_ops()) {
        if (const auto attention = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(node)) {
            const auto& name = attention->get_friendly_name();
            const auto suffix = std::string("/npuw_dense");
            ASSERT_GE(name.size(), suffix.size());
            ASSERT_EQ(name.substr(name.size() - suffix.size()), suffix);
            const auto original = original_masks.find(name.substr(0, name.size() - suffix.size()));
            ASSERT_NE(original, original_masks.end());
            auto old_mask = std::make_shared<v0::Result>(original->second);
            auto new_mask = std::make_shared<v0::Result>(attention->input_value(3));
            old_mask->output(0).set_names({"qsa_old_mask_" + std::to_string(index)});
            new_mask->output(0).set_names({"qsa_new_mask_" + std::to_string(index)});
            extra_results.insert(extra_results.end(), {old_mask, new_mask});
            ++index;
        }
    }
    ASSERT_EQ(index, 8);
    model->add_results(extra_results);
    auto compiled = core.compile_model(model, "CPU", hint::inference_precision(element::f32));
    auto request = compiled.create_infer_request();
    constexpr size_t prompt_length = 15;
    for (const auto& port : compiled.inputs()) {
        auto tensor = request.get_tensor(port);
        std::memset(tensor.data(), 0, tensor.get_byte_size());
        const auto names = port.get_names();
        if (names.count("attention_mask")) {
            std::fill_n(tensor.data<int64_t>() + 64 - prompt_length, prompt_length, 1);
        } else if (names.count("position_ids")) {
            for (size_t plane = 0; plane < 4; ++plane) {
                for (size_t token = 0; token < prompt_length; ++token) {
                    tensor.data<int64_t>()[plane * 64 + 64 - prompt_length + token] = static_cast<int64_t>(token);
                }
            }
        } else if (names.count("inputs_embeds")) {
            for (size_t i = 0; i < tensor.get_size(); ++i) {
                tensor.data<float>()[i] = static_cast<float>(static_cast<int>(i % 31) - 15) * 0.001f;
            }
        }
    }
    request.infer();
    for (size_t layer = 0; layer < 8; ++layer) {
        const auto old_mask = request.get_tensor("qsa_old_mask_" + std::to_string(layer));
        const auto new_mask = request.get_tensor("qsa_new_mask_" + std::to_string(layer));
        ASSERT_EQ(old_mask.get_shape(), (Shape{1, 16, 64, 64}));
        ASSERT_EQ(new_mask.get_shape(), (Shape{64, 64}));
        const auto* old_values = old_mask.data<const uint8_t>();
        const auto* new_values = new_mask.data<const uint8_t>();
        for (size_t head = 0; head < 16; ++head) {
            for (size_t offset = 0; offset < 64 * 64; ++offset) {
                ASSERT_EQ(old_values[head * 64 * 64 + offset], new_values[offset])
                    << "layer " << layer << ", head " << head << ", mask offset " << offset;
            }
        }
    }
}

class QSACompiledModelTest : public ov::test::npuw::LLMPassTestFixture {};

TEST_F(QSACompiledModelTest, RealQwen4IsLoweredBeforeSubmodelCompilation) {
    const auto* path = std::getenv("NPUW_QSA_TEST_MODEL");
    if (!path) {
        GTEST_SKIP() << "Set NPUW_QSA_TEST_MODEL to the Qwen4 language-model XML";
    }
    Core core;
    auto model = core.read_model(path);
    ov::test::npuw::RecordingFactory recorder;
    auto compiled = create_compiled_model(model,
                                          {{"NPUW_LLM_MAX_PROMPT_LEN", "64"},
                                           {"NPUW_LLM_MIN_RESPONSE_LEN", "64"},
                                           {"NPUW_LLM_PREFILL_HINT", "STATIC"},
                                           {"NPUW_LLM_PREFILL_ATTENTION_HINT", "STATIC"},
                                           {"NPUW_LLM_GENERATE_ATTENTION_HINT", "STATIC"},
                                           {"NPUW_LLM_OPTIMIZE_V_TENSORS", "NO"},
                                           {"NPUW_LLM_SHARED_HEAD", "NO"}},
                                          recorder);
    ASSERT_NE(compiled, nullptr);
    const auto& prefill = require_sub_model(recorder, "_prefill");
    const auto& generate = require_sub_model_containing(recorder, "_kv");
    for (const auto* target : {&prefill, &generate}) {
        EXPECT_EQ(target->props.at("NPUW_FOLD").as<std::string>(), "NO");
        EXPECT_EQ(target->props.at("NPUW_FUNCALL_FOR_ALL").as<std::string>(), "NO");
        EXPECT_EQ(target->props.at("NPUW_F16IC").as<std::string>(), "NO");
        EXPECT_EQ(target->props.at("NPUW_ONLINE_PIPELINE").as<std::string>(), "JUST");
    }
    EXPECT_EQ(count_ops<ov::op::v17::SparseAttention>(prefill.model), 0);
    EXPECT_EQ(count_ops<ov::op::v17::SparseAttentionIndexer>(prefill.model), 0);
    EXPECT_EQ(count_ops<ov::op::v17::QSAPrepare>(prefill.model), 0);
    EXPECT_EQ(count_ops<ov::op::v17::SparseAttention>(generate.model), 0);
    EXPECT_EQ(count_ops<ov::op::v17::SparseAttentionIndexer>(generate.model), 0);
    EXPECT_EQ(count_ops<ov::op::v17::QSAPrepare>(generate.model), 0);
    for (const auto& target : {prefill.model, generate.model}) {
        EXPECT_EQ(count_ops<ov::op::v13::ScaledDotProductAttention>(target), 0);
        size_t score_count = 0;
        for (const auto& node : target->get_ordered_ops()) {
            if (is_sparse_score(node)) {
                ++score_count;
                EXPECT_EQ(node->get_output_partial_shape(0).rank().get_length(), 5);
            }
        }
        EXPECT_EQ(score_count, 8u);
    }
    const auto prefill_cache = input_shape(prefill.model, "cache_params.past.block_index_key.0");
    const auto generate_cache = input_shape(generate.model, "cache_params.past.block_index_key.0");
    ASSERT_TRUE(prefill_cache.has_value());
    ASSERT_TRUE(generate_cache.has_value());
    EXPECT_EQ(*prefill_cache, (Shape{1, 16, 128}));
    EXPECT_EQ(*generate_cache, (Shape{1, 32, 128}));
}

TEST_F(QSACompiledModelTest, RealQwen4LongContextUsesCompactAttentionScores) {
    const auto* path = std::getenv("NPUW_QSA_TEST_MODEL");
    if (!path) {
        GTEST_SKIP() << "Set NPUW_QSA_TEST_MODEL to the Qwen4 language-model XML";
    }
    Core core;
    auto model = core.read_model(path);
    ov::test::npuw::RecordingFactory recorder;
    auto compiled = create_compiled_model(model,
                                          {{"NPUW_LLM_MAX_PROMPT_LEN", "64"},
                                           {"NPUW_LLM_MIN_RESPONSE_LEN", "4032"},
                                           {"NPUW_LLM_PREFILL_HINT", "STATIC"},
                                           {"NPUW_LLM_PREFILL_ATTENTION_HINT", "STATIC"},
                                           {"NPUW_LLM_GENERATE_ATTENTION_HINT", "STATIC"},
                                           {"NPUW_LLM_SHARED_HEAD", "NO"}},
                                          recorder);
    ASSERT_NE(compiled, nullptr);
    const auto& generate = require_sub_model_containing(recorder, "_kv");
    EXPECT_EQ(count_ops<ov::op::v13::ScaledDotProductAttention>(generate.model), 0);
    size_t score_count = 0;
    for (const auto& node : generate.model->get_ordered_ops()) {
        if (is_sparse_score(node)) {
            ++score_count;
            const auto shape = node->get_output_partial_shape(0);
            ASSERT_TRUE(shape.is_static());
            ASSERT_EQ(shape.size(), 5u);
            EXPECT_EQ(shape[4].get_length(), 2051);
        }
    }
    EXPECT_EQ(score_count, 8u);
}

TEST_F(QSACompiledModelTest, ExplicitPartitionOptionsOverrideQsaDefaults) {
    const auto* path = std::getenv("NPUW_QSA_TEST_MODEL");
    if (!path) {
        GTEST_SKIP() << "Set NPUW_QSA_TEST_MODEL to the Qwen4 language-model XML";
    }
    Core core;
    auto model = core.read_model(path);
    ov::test::npuw::RecordingFactory recorder;
    auto compiled = create_compiled_model(model,
                                          {{"NPUW_LLM_MAX_PROMPT_LEN", "64"},
                                           {"NPUW_LLM_MIN_RESPONSE_LEN", "64"},
                                           {"NPUW_LLM_PREFILL_HINT", "STATIC"},
                                           {"NPUW_LLM_PREFILL_ATTENTION_HINT", "STATIC"},
                                           {"NPUW_LLM_GENERATE_ATTENTION_HINT", "STATIC"},
                                           {"NPUW_LLM_SHARED_HEAD", "NO"},
                                           {"NPUW_FOLD", "YES"},
                                           {"NPUW_FUNCALL_FOR_ALL", "YES"},
                                           {"NPUW_F16IC", "YES"},
                                           {"NPUW_ONLINE_PIPELINE", "REP"}},
                                          recorder);
    ASSERT_NE(compiled, nullptr);
    const auto& prefill = require_sub_model(recorder, "_prefill");
    const auto& generate = require_sub_model_containing(recorder, "_kv");
    for (const auto* target : {&prefill, &generate}) {
        EXPECT_EQ(target->props.at("NPUW_FOLD").as<std::string>(), "YES");
        EXPECT_EQ(target->props.at("NPUW_FUNCALL_FOR_ALL").as<std::string>(), "YES");
        EXPECT_EQ(target->props.at("NPUW_F16IC").as<std::string>(), "YES");
        EXPECT_EQ(target->props.at("NPUW_ONLINE_PIPELINE").as<std::string>(), "REP");
    }
}

TEST_F(QSACompiledModelTest, SharedHeadUsesQsaPartitionDefaults) {
    const auto* path = std::getenv("NPUW_QSA_TEST_MODEL");
    if (!path) {
        GTEST_SKIP() << "Set NPUW_QSA_TEST_MODEL to the Qwen4 language-model XML";
    }
    Core core;
    auto model = core.read_model(path);
    ov::test::npuw::RecordingFactory recorder;
    auto compiled = create_compiled_model(model,
                                          {{"NPUW_LLM_MAX_PROMPT_LEN", "64"},
                                           {"NPUW_LLM_MIN_RESPONSE_LEN", "64"},
                                           {"NPUW_LLM_PREFILL_HINT", "STATIC"},
                                           {"NPUW_LLM_PREFILL_ATTENTION_HINT", "STATIC"},
                                           {"NPUW_LLM_GENERATE_ATTENTION_HINT", "STATIC"},
                                           {"NPUW_LLM_SHARED_HEAD", "YES"}},
                                          recorder);
    ASSERT_NE(compiled, nullptr);
    const auto& head = require_sub_model(recorder, "_lm_head");
    EXPECT_EQ(head.props.at("NPUW_FOLD").as<std::string>(), "NO");
    EXPECT_EQ(head.props.at("NPUW_FUNCALL_FOR_ALL").as<std::string>(), "NO");
    EXPECT_EQ(head.props.at("NPUW_F16IC").as<std::string>(), "NO");
    EXPECT_EQ(head.props.at("NPUW_ONLINE_PIPELINE").as<std::string>(), "NONE");
}

TEST_F(QSACompiledModelTest, SharedHeadHonorsExplicitPartitionOverrides) {
    const auto* path = std::getenv("NPUW_QSA_TEST_MODEL");
    if (!path) {
        GTEST_SKIP() << "Set NPUW_QSA_TEST_MODEL to the Qwen4 language-model XML";
    }
    Core core;
    auto model = core.read_model(path);
    ov::test::npuw::RecordingFactory recorder;
    auto compiled = create_compiled_model(
        model,
        {{"NPUW_LLM_MAX_PROMPT_LEN", "64"},
         {"NPUW_LLM_MIN_RESPONSE_LEN", "64"},
         {"NPUW_LLM_PREFILL_HINT", "STATIC"},
         {"NPUW_LLM_PREFILL_ATTENTION_HINT", "STATIC"},
         {"NPUW_LLM_GENERATE_ATTENTION_HINT", "STATIC"},
         {"NPUW_LLM_SHARED_HEAD", "YES"},
         {"NPUW_FOLD", "YES"},
         {"NPUW_FUNCALL_FOR_ALL", "YES"},
         {"NPUW_F16IC", "YES"},
         {"++NPUW_LLM_SHARED_HEAD_CONFIG", ov::AnyMap{{"NPUW_FOLD", "NO"}, {"NPUW_F16IC", "NO"}}}},
        recorder);
    ASSERT_NE(compiled, nullptr);
    const auto& head = require_sub_model(recorder, "_lm_head");
    EXPECT_EQ(head.props.at("NPUW_FOLD").as<std::string>(), "NO");
    EXPECT_EQ(head.props.at("NPUW_FUNCALL_FOR_ALL").as<std::string>(), "YES");
    EXPECT_EQ(head.props.at("NPUW_F16IC").as<std::string>(), "NO");
}

}  // namespace
