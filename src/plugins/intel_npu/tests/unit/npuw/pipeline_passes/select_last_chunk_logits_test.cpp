// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/select_last_chunk_logits.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>

#include "openvino/op/ops.hpp"

namespace {

constexpr int64_t kSequenceSize = 32;
constexpr int64_t kSelectedTokenCount = 4;
constexpr int64_t kVocabSize = 2;

std::shared_ptr<ov::Model> build_logits_model(uint32_t batch_dim) {
    const ov::PartialShape logits_shape = batch_dim == 0u ? ov::PartialShape{1, kSequenceSize, kVocabSize}
                                                          : ov::PartialShape{kSequenceSize, 1, kVocabSize};
    auto logits = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, logits_shape);
    logits->output(0).set_names({"logits_input"});

    auto result = std::make_shared<ov::op::v0::Result>(logits);
    result->output(0).set_names({"logits"});
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{logits});
}

void expect_selected_logits(const std::shared_ptr<ov::Model>& model,
                            uint32_t batch_dim,
                            const std::array<int64_t, kSelectedTokenCount>& indices) {
    auto logits = ov::Tensor(ov::element::f32, model->get_parameters().at(0)->get_shape());
    auto* logits_data = logits.data<float>();
    const auto& logits_shape = logits.get_shape();
    const size_t sequence_dim = 1u - batch_dim;
    for (int64_t token_idx = 0; token_idx < kSequenceSize; ++token_idx) {
        size_t offset = 0;
        if (sequence_dim == 0u) {
            offset = static_cast<size_t>(token_idx) * logits_shape[1] * kVocabSize;
        } else {
            offset = static_cast<size_t>(token_idx) * kVocabSize;
        }
        logits_data[offset] = static_cast<float>(100 + token_idx);
        logits_data[offset + 1] = static_cast<float>(200 + token_idx);
    }

    auto selection_indices = ov::Tensor(ov::element::i64, ov::Shape{kSelectedTokenCount});
    std::copy(indices.begin(), indices.end(), selection_indices.data<int64_t>());

    auto output = ov::Tensor(ov::element::f32, model->output().get_shape());
    ASSERT_TRUE(model->evaluate(ov::TensorVector{output}, ov::TensorVector{logits, selection_indices}));
    const ov::Shape expected_shape =
        batch_dim == 0u ? ov::Shape{1, kSelectedTokenCount, kVocabSize} : ov::Shape{kSelectedTokenCount, 1, kVocabSize};
    EXPECT_EQ(output.get_shape(), expected_shape);
    for (size_t index = 0; index < indices.size(); ++index) {
        EXPECT_FLOAT_EQ(output.data<float>()[index * kVocabSize], static_cast<float>(100 + indices[index]));
        EXPECT_FLOAT_EQ(output.data<float>()[index * kVocabSize + 1], static_cast<float>(200 + indices[index]));
    }
}

class SelectLastChunkLogitsTest : public ::testing::TestWithParam<uint32_t> {};

TEST_P(SelectLastChunkLogitsTest, SelectsHostProvidedRowsWithoutMaskAssumptions) {
    const uint32_t batch_dim = GetParam();
    auto model = build_logits_model(batch_dim);

    ASSERT_TRUE(ov::npuw::SelectLastLogits(batch_dim, kSelectedTokenCount, "logits").run_on_model(model));
    const ov::Shape expected_shape =
        batch_dim == 0u ? ov::Shape{1, kSelectedTokenCount, kVocabSize} : ov::Shape{kSelectedTokenCount, 1, kVocabSize};
    EXPECT_EQ(model->output().get_shape(), expected_shape);

    expect_selected_logits(model, batch_dim, {0, 0, 1, 4});
    expect_selected_logits(model, batch_dim, {0, 2, 4, 6});
}

INSTANTIATE_TEST_SUITE_P(BatchDimension, SelectLastChunkLogitsTest, ::testing::Values(0u, 1u));

}  // namespace
