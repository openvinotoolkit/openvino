// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "qwen3asr/prepare_qwen3_asr_model.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "llm_stored_tokens_state.hpp"
#include "llm_test_helpers.hpp"
#include "npuw_transformations/reshape_to_static.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/openvino.hpp"
#include "qwen3asr/qwen3_asr_infer_request.hpp"
#include "util.hpp"

namespace {

std::shared_ptr<ov::Model> make_decoder_mask_model(bool convert_keys = false, size_t branches = 1) {
    using namespace ov::op;
    auto ids = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});
    ids->output(0).set_names({"input_ids"});
    auto past = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, 1});
    past->output(0).set_names({"past_key_values.0.key"});
    auto zero = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto one = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto two = v0::Constant::create(ov::element::i64, ov::Shape{}, {2});
    auto past_length = std::make_shared<v8::Gather>(std::make_shared<v3::ShapeOf>(past), two, zero);
    auto query_length = std::make_shared<v8::Gather>(std::make_shared<v3::ShapeOf>(ids), one, zero);
    auto total_length = std::make_shared<v1::Add>(past_length, query_length);
    auto keys = std::make_shared<v4::Range>(zero, total_length, one, ov::element::i64);
    auto queries = std::make_shared<v4::Range>(past_length, total_length, one, ov::element::i64);
    auto keys_u1 = std::make_shared<v0::Unsqueeze>(keys, zero);
    auto keys_u2 = std::make_shared<v0::Unsqueeze>(keys_u1, zero);
    ov::Output<ov::Node> key_positions = std::make_shared<v0::Unsqueeze>(keys_u2, zero);
    ov::Output<ov::Node> query_positions =
        std::make_shared<v0::Unsqueeze>(queries, v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 1, 3}));
    if (convert_keys) {
        key_positions = std::make_shared<v0::Convert>(key_positions, ov::element::f32);
        query_positions = std::make_shared<v0::Convert>(query_positions, ov::element::f32);
    }

    ov::OutputVector outputs;
    for (size_t branch = 0; branch < branches; ++branch) {
        auto causal = std::make_shared<v1::LessEqual>(key_positions, query_positions);
        auto scores = std::make_shared<v1::Select>(causal,
                                                   v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f}),
                                                   v0::Constant::create(ov::element::f32, ov::Shape{}, {-1.0e9f}));
        outputs.push_back(std::make_shared<v8::Softmax>(scores, -1));
        outputs.push_back(
            std::make_shared<v1::Reshape>(queries,
                                          v0::Constant::create(ov::element::i64, ov::Shape{3}, {1, 1, -1}),
                                          false));
    }
    return std::make_shared<ov::Model>(outputs, ov::ParameterVector{ids, past});
}

void make_static(const std::shared_ptr<ov::Model>& model, size_t query_length, size_t past_length) {
    std::map<std::string, ov::PartialShape> shapes{
        {"input_ids", {1, static_cast<int64_t>(query_length)}},
        {"past_key_values.0.key", {1, 1, static_cast<int64_t>(past_length), 1}},
    };
    for (const auto& input : model->inputs()) {
        if (input.get_names().count("attention_mask")) {
            shapes["attention_mask"] = {1, static_cast<int64_t>(query_length + past_length)};
        }
        if (input.get_names().count("position_ids")) {
            shapes["position_ids"] = {static_cast<int64_t>(query_length)};
        }
    }
    model->reshape(shapes);
}

ov::TensorVector evaluate(const std::shared_ptr<ov::Model>& model,
                          const std::map<std::string, std::vector<int64_t>>& values = {}) {
    ov::TensorVector inputs;
    for (const auto& input : model->inputs()) {
        ov::Tensor tensor(input.get_element_type(), input.get_shape());
        std::memset(tensor.data(), 0, tensor.get_byte_size());
        const auto it = values.find(input.get_any_name());
        if (it != values.end()) {
            OPENVINO_ASSERT(tensor.get_size() == it->second.size());
            std::copy(it->second.begin(), it->second.end(), tensor.data<int64_t>());
        }
        inputs.push_back(tensor);
    }
    ov::TensorVector outputs;
    for (const auto& output : model->outputs()) {
        outputs.emplace_back(output.get_element_type(), output.get_shape());
    }
    OPENVINO_ASSERT(model->evaluate(outputs, inputs), "Qwen3-ASR reference evaluation failed");
    return outputs;
}

size_t count_inputs(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    const auto inputs = model->inputs();
    return std::count_if(inputs.begin(), inputs.end(), [&](const auto& input) {
        return input.get_names().count(name) != 0;
    });
}

class PrepareQwen3ASRMaskTest : public ::testing::TestWithParam<bool> {};

TEST_P(PrepareQwen3ASRMaskTest, StaticGenerationMasksUnusedSlotsAndKeepsAbsolutePositions) {
    constexpr size_t capacity = 1024;
    auto original = make_decoder_mask_model(GetParam());
    make_static(original, 1, capacity - 1);
    const auto unmasked = evaluate(original);
    const auto* original_weights = unmasked[0].data<const float>();
    EXPECT_GT(std::accumulate(original_weights + 53, original_weights + capacity - 1, 0.0f), 0.94f);
    EXPECT_EQ(unmasked[1].data<const int64_t>()[0], capacity - 1);

    auto model = make_decoder_mask_model(GetParam());
    ov::npuw::PrepareQwen3ASRKVCacheModel().run_on_model(model);
    make_static(model, 1, capacity - 1);
    for (const size_t stored : {1u, 53u, 54u, 1023u}) {
        std::vector<int64_t> mask(capacity, 1);
        std::fill_n(mask.begin(), stored, 0);
        mask.back() = 0;
        const auto outputs = evaluate(model, {{"attention_mask", mask}, {"position_ids", {int64_t(stored)}}});
        const auto* weights = outputs[0].data<const float>();
        EXPECT_FLOAT_EQ(std::accumulate(weights + stored, weights + capacity - 1, 0.0f), 0.0f);
        EXPECT_NEAR(std::accumulate(weights, weights + capacity, 0.0f), 1.0f, 1.0e-5f);
        EXPECT_NEAR(weights[capacity - 1], 1.0f / (stored + 1), 1.0e-6f);
        EXPECT_EQ(outputs[1].data<const int64_t>()[0], stored);
    }
}

TEST_P(PrepareQwen3ASRMaskTest, PrefillPreservesCausalityAndExcludesLeftPadding) {
    constexpr size_t capacity = 16;
    constexpr size_t padding = 13;
    auto model = make_decoder_mask_model(GetParam());
    ov::npuw::PrepareQwen3ASRPrefillModel().run_on_model(model);
    make_static(model, capacity, 0);
    std::vector<int64_t> mask(capacity, 1);
    std::fill(mask.begin() + padding, mask.end(), 0);
    std::vector<int64_t> positions(capacity, 0);
    std::iota(positions.begin() + padding, positions.end(), 0);
    const auto outputs = evaluate(model, {{"attention_mask", mask}, {"position_ids", positions}});
    const auto* weights = outputs[0].data<const float>();
    for (size_t query = padding; query < capacity; ++query) {
        for (size_t key = 0; key < capacity; ++key) {
            const float expected = key >= padding && key <= query ? 1.0f / (query - padding + 1) : 0.0f;
            EXPECT_NEAR(weights[query * capacity + key], expected, 1.0e-6f);
        }
        EXPECT_EQ(outputs[1].data<const int64_t>()[query], query - padding);
    }
}

TEST(PrepareQwen3ASRModelTest, SharesInputsAcrossMultipleAttentionAndPositionBranches) {
    for (const bool prefill : {false, true}) {
        auto model = make_decoder_mask_model(false, 2);
        if (prefill) {
            ov::npuw::PrepareQwen3ASRPrefillModel().run_on_model(model);
        } else {
            ov::npuw::PrepareQwen3ASRKVCacheModel().run_on_model(model);
        }
        EXPECT_EQ(count_inputs(model, "attention_mask"), 1u);
        EXPECT_EQ(count_inputs(model, "position_ids"), 1u);
    }
}

TEST(PrepareQwen3ASRModelTest, CanonicalKVShapesStayIndependentOfEncoderLength) {
    constexpr uint32_t capacity = 128;
    constexpr uint32_t encoder_length = 64;
    const ov::npuw::KVAxesPosition axes{0, 2};
    for (const bool prefill : {true, false}) {
        SCOPED_TRACE(prefill);
        auto model = make_decoder_mask_model();
        if (prefill) {
            ov::npuw::PrepareQwen3ASRPrefillModel().run_on_model(model);
        } else {
            ov::npuw::PrepareQwen3ASRKVCacheModel().run_on_model(model);
        }
        const uint32_t input_size = prefill ? capacity : 1;
        ov::npuw::ReshapeToStatic(input_size, capacity, axes, 0, encoder_length, prefill, false).run_on_model(model);

        EXPECT_EQ(model->input("past_key_values.0.key").get_shape(), (ov::Shape{1, 1, capacity - input_size, 1}));
        EXPECT_EQ(model->input("attention_mask").get_shape(), (ov::Shape{1, capacity}));
        EXPECT_EQ(model->input("position_ids").get_shape(), (ov::Shape{input_size}));
        EXPECT_EQ(model->output(0).get_shape(), (ov::Shape{1, 1, input_size, capacity}));
    }
}

TEST(PrepareQwen3ASRModelTest, WhisperCacheShapesStillUseEncoderLength) {
    constexpr uint32_t capacity = 128;
    constexpr uint32_t encoder_length = 64;
    const ov::npuw::KVAxesPosition axes{0, 2};
    for (const bool prefill : {true, false}) {
        SCOPED_TRACE(prefill);
        const uint32_t input_size = prefill ? capacity : 1;
        auto encoder_cache = make_decoder_mask_model();
        ov::npuw::ReshapeToStatic(input_size, capacity, axes, 0, encoder_length, prefill, true)
            .run_on_model(encoder_cache);
        EXPECT_EQ(encoder_cache->input("past_key_values.0.key").get_shape(), (ov::Shape{1, 1, encoder_length, 1}));

        auto decoder_cache = make_decoder_mask_model();
        decoder_cache->input("past_key_values.0.key").get_tensor().set_names({"past_key_values.0.decoder.key"});
        ov::npuw::ReshapeToStatic(input_size, capacity, axes, 0, encoder_length, prefill, true)
            .run_on_model(decoder_cache);
        EXPECT_EQ(decoder_cache->input("past_key_values.0.decoder.key").get_shape(),
                  (ov::Shape{1, 1, capacity - input_size, 1}));
    }
}

TEST(Qwen3ASRInferRequestTest, ResetRestartsSingleTokenPrefillAndPublishesState) {
    auto model = ov::test::npuw::build_llm_test_model();
    auto encoder = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 64});
    encoder->output(0).set_names({"encoder_hidden_states"});
    model->add_parameters({encoder});
    auto plugin = std::make_shared<ov::test::npuw::NullPlugin>();
    ov::test::npuw::RecordingFactory recorder;
    ov::AnyMap properties{{"NPUW_LLM", "YES"},
                          {"NPUW_QWEN3_ASR", "YES"},
                          {"NPUW_QWEN3_ASR_MAX_ENCODER_LEN", "16"},
                          {"NPUW_LLM_MAX_PROMPT_LEN", "64"},
                          {"NPUW_LLM_MIN_RESPONSE_LEN", "64"},
                          {"NPUW_LLM_PREFILL_HINT", "STATIC"},
                          {"NPUW_DEVICES", "CPU"}};
    auto compiled = std::make_shared<ov::npuw::LLMCompiledModel>(model, plugin, properties, recorder.make_factory());
    ov::npuw::Qwen3ASRInferRequest request(compiled);

    const auto set_input =
        [](ov::npuw::Qwen3ASRInferRequest& target, const std::string& name, const ov::Tensor& tensor) {
            const auto port = ov::npuw::util::find_port_by_name(target.get_inputs(), name);
            ASSERT_TRUE(port.has_value()) << "Missing input " << name;
            target.set_tensor(*port, ov::get_tensor_impl(tensor));
        };
    ov::Tensor encoder_hidden_states(ov::element::f32, {1, 1, 64});
    std::fill_n(encoder_hidden_states.data<float>(), encoder_hidden_states.get_size(), 1.0f);
    ov::Tensor one_token(ov::element::i64, {1, 1});
    one_token.data<int64_t>()[0] = 1;
    ov::npuw::Qwen3ASRInferRequest fresh(compiled);
    set_input(fresh, "encoder_hidden_states", encoder_hidden_states);
    set_input(fresh, "input_ids", one_token);
    ASSERT_NO_THROW(fresh.infer());
    const auto fresh_logits_port = ov::npuw::util::find_port_by_name(fresh.get_outputs(), "logits");
    ASSERT_TRUE(fresh_logits_port.has_value());
    auto fresh_logits = fresh.get_tensor(*fresh_logits_port);
    ov::Tensor expected_logits(fresh_logits->get_element_type(), fresh_logits->get_shape());
    std::memcpy(expected_logits.data(), fresh_logits->data(), fresh_logits->get_byte_size());

    set_input(request, "encoder_hidden_states", encoder_hidden_states);
    ov::Tensor prompt(ov::element::i64, {1, 2});
    std::fill_n(prompt.data<int64_t>(), prompt.get_size(), 1);
    set_input(request, "input_ids", prompt);
    ASSERT_NO_THROW(request.infer());

    const auto states = request.query_state();
    const auto state = std::find_if(states.begin(), states.end(), [](const auto& candidate) {
        return candidate->get_name() == ov::npuw::StoredTokensState::state_name;
    });
    ASSERT_NE(state, states.end());
    EXPECT_EQ((*state)->get_state()->data<int64_t>()[0], 2);

    set_input(request, "input_ids", one_token);
    ASSERT_NO_THROW(request.infer());
    EXPECT_EQ((*state)->get_state()->data<int64_t>()[0], 3);

    ASSERT_NO_THROW(fresh.infer());
    const auto fresh_states = fresh.query_state();
    const auto fresh_state = std::find_if(fresh_states.begin(), fresh_states.end(), [](const auto& candidate) {
        return candidate->get_name() == ov::npuw::StoredTokensState::state_name;
    });
    ASSERT_NE(fresh_state, fresh_states.end());
    EXPECT_EQ((*fresh_state)->get_state()->data<int64_t>()[0], 2);
    EXPECT_EQ((*state)->get_state()->data<int64_t>()[0], 3);

    (*state)->reset();
    EXPECT_EQ((*state)->get_state()->data<int64_t>()[0], 0);
    ASSERT_NO_THROW(request.infer());
    EXPECT_EQ((*state)->get_state()->data<int64_t>()[0], 1);
    const auto logits_port = ov::npuw::util::find_port_by_name(request.get_outputs(), "logits");
    ASSERT_TRUE(logits_port.has_value());
    const auto actual_logits = request.get_tensor(*logits_port);
    ASSERT_EQ(actual_logits->get_shape(), expected_logits.get_shape());
    ASSERT_EQ(actual_logits->get_byte_size(), expected_logits.get_byte_size());
    EXPECT_EQ(std::memcmp(actual_logits->data(), expected_logits.data(), expected_logits.get_byte_size()), 0);
}

TEST(PrepareQwen3ASRModelTest, RejectsUnsupportedExportInsteadOfSilentlyFreezingAttention) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});
    input->output(0).set_names({"input_ids"});
    auto model = std::make_shared<ov::Model>(ov::OutputVector{input}, ov::ParameterVector{input});
    EXPECT_THROW(ov::npuw::PrepareQwen3ASRKVCacheModel().run_on_model(model), ov::Exception);
    EXPECT_THROW(ov::npuw::PrepareQwen3ASRPrefillModel().run_on_model(model), ov::Exception);
}

INSTANTIATE_TEST_SUITE_P(ExportedMaskTypes, PrepareQwen3ASRMaskTest, ::testing::Bool());

}  // namespace
