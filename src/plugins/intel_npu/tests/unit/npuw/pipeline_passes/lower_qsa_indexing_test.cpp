// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/lower_qsa_indexing.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include "npuw_transformations/lower_sparse_attention.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/qsa_prepare.hpp"
#include "openvino/op/sparse_attention.hpp"
#include "openvino/op/sparse_attention_indexer.hpp"
#include "openvino/openvino.hpp"

namespace {

using namespace ov;
namespace v0 = ov::op::v0;
namespace v17 = ov::op::v17;

std::shared_ptr<v0::Parameter> parameter(const std::string& name, const element::Type& type, const Shape& shape) {
    auto node = std::make_shared<v0::Parameter>(type, shape);
    node->output(0).set_names({name});
    return node;
}

void check_prepare(size_t query_length, const std::vector<int64_t>& key_lengths) {
    constexpr size_t block_size = 4;
    constexpr size_t capacity = 4;
    constexpr size_t dim = 4;
    constexpr size_t rotary = 2;
    constexpr float epsilon = 1e-6f;
    auto raw = parameter("raw", element::f32, {1, query_length, dim});
    auto blocks = parameter("blocks", element::f32, {1, capacity, dim});
    auto length = parameter("length", element::i64, {1});
    auto cos = parameter("cos", element::f32, {1, query_length, rotary});
    auto sin = parameter("sin", element::f32, {1, query_length, rotary});
    auto weight = v0::Constant::create(element::f32, Shape{dim}, {1.f, 1.1f, 0.9f, 1.2f});
    auto incomplete = parameter("incomplete", element::f32, {1, block_size - 1, dim});
    auto rope = parameter("rope", element::f32, {1, rotary * 2});
    auto prepare =
        std::make_shared<v17::QSAPrepare>(raw, blocks, length, cos, sin, weight, incomplete, rope, block_size, epsilon);
    auto model = std::make_shared<Model>(OutputVector{prepare->output(0), prepare->output(1), prepare->output(2)},
                                         ParameterVector{raw, blocks, length, cos, sin, incomplete, rope});
    ASSERT_TRUE(ov::npuw::LowerQSAPrepare().run_on_model(model));
    model->validate_nodes_and_infer_types();
    Core core;
    auto request = core.compile_model(model, "CPU", hint::inference_precision(element::f32)).create_infer_request();
    auto raw_tensor = request.get_tensor("raw");
    auto block_tensor = request.get_tensor("blocks");
    auto len_tensor = request.get_tensor("length");
    auto cos_tensor = request.get_tensor("cos");
    auto sin_tensor = request.get_tensor("sin");
    auto incomplete_tensor = request.get_tensor("incomplete");
    auto rope_tensor = request.get_tensor("rope");
    using RopeRow = std::array<float, rotary * 2>;
    for (size_t i = 0; i < raw_tensor.get_size(); ++i) {
        raw_tensor.data<float>()[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.07f;
    }
    for (size_t i = 0; i < block_tensor.get_size(); ++i) {
        block_tensor.data<float>()[i] = static_cast<float>(i + 1) * 0.03f;
    }
    for (size_t i = 0; i < incomplete_tensor.get_size(); ++i) {
        incomplete_tensor.data<float>()[i] = static_cast<float>(static_cast<int>(i) - 5) * 0.04f;
    }
    for (size_t row = 0; row < query_length; ++row) {
        const auto angle = static_cast<float>(row + 1) * 0.13f;
        std::fill_n(cos_tensor.data<float>() + row * rotary, rotary, std::cos(angle));
        std::fill_n(sin_tensor.data<float>() + row * rotary, rotary, std::sin(angle));
    }
    std::copy_n(std::array<float, 4>{0.8f, 0.8f, 0.6f, 0.6f}.data(), 4, rope_tensor.data<float>());
    const std::array<float, dim> gamma{1.f, 1.1f, 0.9f, 1.2f};

    for (const auto key_length : key_lengths) {
        SCOPED_TRACE(key_length);
        len_tensor.data<int64_t>()[0] = key_length;
        request.infer();
        const auto past = std::max<int64_t>(0, key_length - static_cast<int64_t>(query_length));
        const auto cached = static_cast<size_t>(past / block_size);
        const auto pending = static_cast<size_t>(past % block_size);
        const auto current = static_cast<size_t>(key_length - past);
        const auto padding = query_length - current;
        std::vector<std::array<float, dim>> keys;
        for (size_t row = 0; row < pending; ++row) {
            std::array<float, dim> key{};
            std::copy_n(incomplete_tensor.data<const float>() + row * dim, dim, key.begin());
            keys.push_back(key);
        }
        for (size_t row = padding; row < query_length; ++row) {
            std::array<float, dim> key{};
            std::copy_n(raw_tensor.data<const float>() + row * dim, dim, key.begin());
            keys.push_back(key);
        }
        auto rope_row = [&](size_t position) {
            RopeRow result{};
            if (position < pending) {
                std::copy_n(rope_tensor.data<const float>(), result.size(), result.begin());
            } else {
                const auto row = padding + position - pending;
                std::copy_n(cos_tensor.data<const float>() + row * rotary, rotary, result.begin());
                std::copy_n(sin_tensor.data<const float>() + row * rotary, rotary, result.begin() + rotary);
            }
            return result;
        };
        std::vector<float> expected_blocks(capacity * dim, 0.f);
        std::copy_n(block_tensor.data<const float>(), cached * dim, expected_blocks.begin());
        const auto complete = keys.size() / block_size;
        for (size_t block = 0; block < complete; ++block) {
            std::array<float, dim> pooled{};
            for (size_t row = 0; row < block_size; ++row) {
                for (size_t d = 0; d < dim; ++d) {
                    pooled[d] += keys[block * block_size + row][d] / block_size;
                }
            }
            float mean_square = 0.f;
            for (const auto x : pooled) {
                mean_square += x * x / dim;
            }
            for (size_t d = 0; d < dim; ++d) {
                pooled[d] = pooled[d] / std::sqrt(mean_square + epsilon) * gamma[d];
            }
            const auto coefficients = rope_row(block * block_size);
            const auto x0 = pooled[0];
            const auto x1 = pooled[1];
            pooled[0] = x0 * coefficients[0] - x1 * coefficients[rotary];
            pooled[1] = x1 * coefficients[1] + x0 * coefficients[rotary + 1];
            std::copy(pooled.begin(), pooled.end(), expected_blocks.begin() + (cached + block) * dim);
        }
        std::vector<float> expected_tail((block_size - 1) * dim, 0.f);
        const auto rest = keys.size() - complete * block_size;
        for (size_t row = 0; row < rest; ++row) {
            std::copy(keys[complete * block_size + row].begin(),
                      keys[complete * block_size + row].end(),
                      expected_tail.begin() + row * dim);
        }
        RopeRow expected_rope{};
        if (rest) {
            expected_rope = rope_row(complete * block_size);
        }
        const auto actual_blocks = request.get_output_tensor(0);
        const auto actual_tail = request.get_output_tensor(1);
        const auto actual_rope = request.get_output_tensor(2);
        ASSERT_EQ(actual_blocks.get_shape(), (Shape{1, capacity, dim}));
        for (size_t i = 0; i < expected_blocks.size(); ++i) {
            EXPECT_NEAR(actual_blocks.data<const float>()[i], expected_blocks[i], 2e-5f) << "block element " << i;
        }
        for (size_t i = 0; i < expected_tail.size(); ++i) {
            EXPECT_NEAR(actual_tail.data<const float>()[i], expected_tail[i], 2e-5f) << "tail element " << i;
        }
        for (size_t i = 0; i < expected_rope.size(); ++i) {
            EXPECT_NEAR(actual_rope.data<const float>()[i], expected_rope[i], 2e-5f) << "rope element " << i;
        }
    }
}

TEST(LowerQSAPrepareTest, PreservesPaddedPrefillAndIncompleteBlocks) {
    check_prepare(5, {0, 1, 2, 3, 4, 5});
}

TEST(LowerQSAPrepareTest, PreservesDecodeAcrossBlockBoundaries) {
    check_prepare(1, {1, 2, 3, 4, 5, 6, 7, 8, 12, 16});
}

TEST(LowerQSAPrepareTest, RejectsUnsupportedPrecision) {
    auto raw = parameter("raw", element::f16, {1, 1, 4});
    auto blocks = parameter("blocks", element::f16, {1, 4, 4});
    auto length = parameter("length", element::i64, {1});
    auto cos = parameter("cos", element::f16, {1, 1, 2});
    auto sin = parameter("sin", element::f16, {1, 1, 2});
    auto weights = parameter("weights", element::f16, {4});
    auto incomplete = parameter("incomplete", element::f16, {1, 3, 4});
    auto rope = parameter("rope", element::f16, {1, 4});
    auto prepare =
        std::make_shared<v17::QSAPrepare>(raw, blocks, length, cos, sin, weights, incomplete, rope, 4, 1e-6f);
    auto model = std::make_shared<Model>(OutputVector{prepare->output(0), prepare->output(1), prepare->output(2)},
                                         ParameterVector{raw, blocks, length, cos, sin, weights, incomplete, rope});
    EXPECT_THROW(ov::npuw::LowerQSAPrepare().run_on_model(model), ov::Exception);
    EXPECT_TRUE(ov::is_type<v17::QSAPrepare>(model->get_results()[0]->input_value(0).get_node_shared_ptr()));
}

void check_indexer(size_t width) {
    constexpr size_t heads = 2;
    constexpr size_t queries = 3;
    constexpr size_t dim = 2;
    constexpr size_t capacity = 4;
    auto query = parameter("query", element::f32, {1, heads, queries, dim});
    auto blocks = parameter("blocks", element::f32, {1, capacity, dim});
    auto length = parameter("length", element::i64, {1});
    auto indexer =
        std::make_shared<v17::SparseAttentionIndexer>(query, blocks, length, 2, static_cast<int64_t>(width * 2), 1);
    auto model = std::make_shared<Model>(OutputVector{indexer->output(0), indexer->output(1)},
                                         ParameterVector{query, blocks, length});
    ASSERT_TRUE(ov::npuw::LowerSparseAttentionIndexer().run_on_model(model));
    model->validate_nodes_and_infer_types();
    Core core;
    auto request = core.compile_model(model, "CPU", hint::inference_precision(element::f32)).create_infer_request();
    auto q = request.get_tensor("query");
    auto k = request.get_tensor("blocks");
    auto len = request.get_tensor("length");
    for (size_t h = 0; h < heads; ++h) {
        for (size_t row = 0; row < queries; ++row) {
            q.data<float>()[(h * queries + row) * dim] = static_cast<float>((h + 1) * (row + 1));
            q.data<float>()[(h * queries + row) * dim + 1] = -0.5f * static_cast<float>(h + 1);
        }
    }
    const std::array<float, capacity * dim> key_values{1.f, 0.f, 1.f, 0.f, 0.f, 1.f, 1.f, 1.f};
    std::copy(key_values.begin(), key_values.end(), k.data<float>());
    for (const int64_t key_length : {0, 1, 3, 5, 7, 8}) {
        SCOPED_TRACE(key_length);
        len.data<int64_t>()[0] = key_length;
        request.infer();
        const auto actual_indices = request.get_output_tensor(0);
        const auto actual_counts = request.get_output_tensor(1);
        ASSERT_EQ(actual_indices.get_shape(), (Shape{1, 1, queries, width}));
        for (size_t row = 0; row < queries; ++row) {
            const auto position = key_length - static_cast<int64_t>(queries) + static_cast<int64_t>(row);
            const auto visible = static_cast<size_t>(std::max<int64_t>(0, (position + 1) / 2));
            std::vector<std::pair<float, int32_t>> scores;
            for (size_t block = 0; block < visible; ++block) {
                float score = 0.f;
                for (size_t h = 0; h < heads; ++h) {
                    const auto* query_row = q.data<const float>() + (h * queries + row) * dim;
                    const auto* key = k.data<const float>() + block * dim;
                    score += std::max(query_row[0] * key[0] + query_row[1] * key[1], 0.f);
                }
                scores.emplace_back(score, static_cast<int32_t>(block));
            }
            std::stable_sort(scores.begin(), scores.end(), [](const auto& a, const auto& b) {
                return a.first > b.first;
            });
            ASSERT_EQ(actual_counts.data<const int32_t>()[row], static_cast<int32_t>(std::min(scores.size(), width)));
            for (size_t slot = 0; slot < width; ++slot) {
                const auto expected = slot < scores.size() ? scores[slot].second : -1;
                EXPECT_EQ(actual_indices.data<const int32_t>()[row * width + slot], expected)
                    << "query " << row << ", slot " << slot;
            }
        }
    }
}

TEST(LowerSparseAttentionIndexerTest, SelectsVisibleBlocksWithStableTiesAndPadding) {
    check_indexer(6);
    check_indexer(2);
}

TEST(LowerSparseAttentionIndexerTest, RejectsWeightedSelection) {
    auto query = parameter("query", element::f32, {1, 1, 1, 1});
    auto blocks = parameter("blocks", element::f32, {1, 1, 1});
    auto length = parameter("length", element::i64, {1});
    auto weights = parameter("weights", element::f32, {1, 1, 1});
    auto indexer = std::make_shared<v17::SparseAttentionIndexer>(query, blocks, length, weights, 4, 4, 1);
    auto model = std::make_shared<Model>(OutputVector{indexer->output(0), indexer->output(1)},
                                         ParameterVector{query, blocks, length, weights});
    EXPECT_THROW(ov::npuw::LowerSparseAttentionIndexer().run_on_model(model), ov::Exception);
    EXPECT_TRUE(
        ov::is_type<v17::SparseAttentionIndexer>(model->get_results()[0]->input_value(0).get_node_shared_ptr()));
}

TEST(LowerSparseAttentionIndexerTest, TokenBudget2048ControlsDenseAttentionMask) {
    constexpr size_t block_size = 4;
    constexpr size_t blocks = 514;
    constexpr size_t keys = blocks * block_size;
    auto query = v0::Constant::create(element::f32, Shape{1, 1, 1, 1}, {0.f});
    auto key = v0::Constant::create(element::f32, Shape{1, 1, keys, 1}, {0.f});
    std::vector<float> values(keys);
    std::iota(values.begin(), values.end(), 1.f);
    auto value = v0::Constant::create(element::f32, Shape{1, 1, keys, 1}, values);
    auto index_query = v0::Constant::create(element::f32, Shape{1, 1, 1, 1}, {1.f});
    std::vector<float> block_scores(blocks);
    std::iota(block_scores.begin(), block_scores.end(), 0.f);
    auto block_keys = v0::Constant::create(element::f32, Shape{1, blocks, 1}, block_scores);
    auto length = v0::Constant::create(element::i64, Shape{1}, {keys});
    auto indexer = std::make_shared<v17::SparseAttentionIndexer>(index_query, block_keys, length, block_size, 2048, 1);
    auto sparse = std::make_shared<v17::SparseAttention>(query,
                                                         key,
                                                         value,
                                                         indexer->output(0),
                                                         indexer->output(1),
                                                         length,
                                                         block_size,
                                                         1,
                                                         true);
    auto model = std::make_shared<Model>(OutputVector{sparse}, ParameterVector{});
    ASSERT_TRUE(ov::npuw::LowerSparseAttention().run_on_model(model));
    ASSERT_TRUE(ov::npuw::LowerSparseAttentionIndexer().run_on_model(model));
    model->validate_nodes_and_infer_types();
    Core core;
    auto request = core.compile_model(model, "CPU", hint::inference_precision(element::f32)).create_infer_request();
    request.infer();
    EXPECT_NEAR(request.get_output_tensor().data<const float>()[0], (9.f + 2056.f) / 2.f, 1e-3f);
}

}  // namespace
