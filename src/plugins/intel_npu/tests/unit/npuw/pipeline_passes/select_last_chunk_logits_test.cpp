// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/select_last_chunk_logits.hpp"

#include <gtest/gtest.h>

#include <numeric>
#include <vector>

#include "openvino/op/ops.hpp"

namespace {

constexpr int64_t kChunkSize = 32;
constexpr int64_t kMaskSize = 96;
constexpr int64_t kVocabSize = 2;

std::shared_ptr<ov::Model> build_logits_model(uint32_t batch_dim) {
    const ov::PartialShape logits_shape =
        batch_dim == 0u ? ov::PartialShape{1, kChunkSize, kVocabSize} : ov::PartialShape{kChunkSize, 1, kVocabSize};
    auto logits = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, logits_shape);
    auto attention_mask = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, kMaskSize});
    logits->output(0).set_names({"logits_input"});
    attention_mask->output(0).set_names({"attention_mask"});

    auto result = std::make_shared<ov::op::v0::Result>(logits);
    result->output(0).set_names({"logits"});
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{logits, attention_mask});
}

void expect_last_token_logits(const std::shared_ptr<ov::Model>& model,
                              int64_t total_token_count,
                              int64_t expected_chunk_index) {
    auto logits = ov::Tensor(ov::element::f32, model->get_parameters().at(0)->get_shape());
    auto* logits_data = logits.data<float>();
    for (int64_t token_idx = 0; token_idx < kChunkSize; ++token_idx) {
        logits_data[token_idx * kVocabSize] = static_cast<float>(100 + token_idx);
        logits_data[token_idx * kVocabSize + 1] = static_cast<float>(200 + token_idx);
    }

    auto attention_mask = ov::Tensor(ov::element::i64, model->get_parameters().at(1)->get_shape());
    std::fill(attention_mask.data<int64_t>(), attention_mask.data<int64_t>() + kMaskSize, 0);
    std::fill(attention_mask.data<int64_t>(), attention_mask.data<int64_t>() + total_token_count, 1);

    auto output = ov::Tensor(ov::element::f32, model->output().get_shape());
    ASSERT_TRUE(model->evaluate(ov::TensorVector{output}, ov::TensorVector{logits, attention_mask}));
    EXPECT_EQ(output.get_shape(), (ov::Shape{1, 1, kVocabSize}));
    EXPECT_FLOAT_EQ(output.data<float>()[0], static_cast<float>(100 + expected_chunk_index));
    EXPECT_FLOAT_EQ(output.data<float>()[1], static_cast<float>(200 + expected_chunk_index));
}

class SelectLastChunkLogitsTest : public ::testing::TestWithParam<uint32_t> {};

TEST_P(SelectLastChunkLogitsTest, SelectsLastTokenFromFullAndLeftAlignedTailChunks) {
    const uint32_t batch_dim = GetParam();
    auto model = build_logits_model(batch_dim);

    ASSERT_TRUE(ov::npuw::SelectLastChunkLogits(batch_dim, kChunkSize).run_on_model(model));
    EXPECT_EQ(model->output().get_shape(), (ov::Shape{1, 1, kVocabSize}));

    expect_last_token_logits(model, 96, 31);
    expect_last_token_logits(model, 65, 0);
}

INSTANTIATE_TEST_SUITE_P(BatchDimension, SelectLastChunkLogitsTest, ::testing::Values(0u, 1u));

}  // namespace