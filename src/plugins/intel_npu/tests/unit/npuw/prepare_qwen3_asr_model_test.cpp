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

#include "openvino/op/ops.hpp"
#include "openvino/openvino.hpp"

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

TEST(PrepareQwen3ASRModelTest, RejectsUnsupportedExportInsteadOfSilentlyFreezingAttention) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});
    input->output(0).set_names({"input_ids"});
    auto model = std::make_shared<ov::Model>(ov::OutputVector{input}, ov::ParameterVector{input});
    EXPECT_THROW(ov::npuw::PrepareQwen3ASRKVCacheModel().run_on_model(model), ov::Exception);
    EXPECT_THROW(ov::npuw::PrepareQwen3ASRPrefillModel().run_on_model(model), ov::Exception);
}

INSTANTIATE_TEST_SUITE_P(ExportedMaskTypes, PrepareQwen3ASRMaskTest, ::testing::Bool());

}  // namespace
