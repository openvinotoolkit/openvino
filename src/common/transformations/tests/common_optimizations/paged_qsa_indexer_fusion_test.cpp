// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/paged_qsa_indexer_fusion.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/paged_qsa_indexer.hpp"
#include "openvino/op/paged_sparse_attention.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/qsa_indexer.hpp"
#include "openvino/op/qsa_prepare.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/sparse_attention.hpp"
#include "openvino/op/sparse_attention_indexer.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"
#include "ov_ops/rms.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "transformations/common_optimizations/fuse_qsa_indexer.hpp"
#include "transformations/common_optimizations/fuse_rotary_positional_embeddings.hpp"
#include "transformations/common_optimizations/rms_fusion.hpp"
#include "transformations/utils/utils.hpp"

namespace {

using namespace ov;
namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v6 = ov::op::v6;
namespace v8 = ov::op::v8;
namespace v13 = ov::op::v13;
namespace v17 = ov::op::v17;
namespace op_internal = ov::op::internal;
using QSAIndexerInputIdx = op_internal::QSAIndexer::InputIdx;
using QSAIndexerOutputIdx = op_internal::QSAIndexer::OutputIdx;
using PagedQSAIndexerInputIdx = op_internal::PagedQSAIndexer::InputIdx;
using PagedQSAIndexerOutputIdx = op_internal::PagedQSAIndexer::OutputIdx;

constexpr int64_t query_heads = 4;
constexpr int64_t key_heads = 1;
constexpr int64_t kv_heads = 2;
constexpr int64_t head_size = 16;
constexpr int64_t rotary_dim = 8;
constexpr int64_t key_block_size = 4;
constexpr int64_t block_topk = 2;
constexpr int64_t selection_heads = 1;
constexpr float epsilon = 1e-6f;

Output<Node> i64(const std::vector<int64_t>& values) {
    return v0::Constant::create(element::i64, Shape{values.size()}, values);
}

std::shared_ptr<v0::Parameter> make_named_param(const std::string& name,
                                                element::Type type,
                                                const PartialShape& shape) {
    const auto param = std::make_shared<v0::Parameter>(type, shape);
    param->set_friendly_name(name);
    param->get_output_tensor(0).set_names({name});
    return param;
}

struct GraphOptions {
    bool query_norm_rope = true;
    bool gathered_states = false;
    bool variable_states = true;
};

// The dense QSAIndexer as created by FuseQSAIndexer, with its states kept in variables.
std::shared_ptr<Model> make_model(const GraphOptions& o, bool paged) {
    ParameterVector params;
    const auto param = [&params](element::Type type, const PartialShape& shape) {
        params.push_back(std::make_shared<v0::Parameter>(type, shape));
        return params.back();
    };
    const auto query = param(element::f32, PartialShape{-1, -1, query_heads, head_size});
    const auto raw_keys = param(element::f32, PartialShape{-1, -1, key_heads, head_size});
    const auto rope_cos = param(element::f32, PartialShape{-1, -1, rotary_dim});
    const auto rope_sin = param(element::f32, PartialShape{-1, -1, rotary_dim});
    const auto key_length = param(element::i32, PartialShape{-1});
    const auto cache_padding = param(element::f32, PartialShape{-1, key_heads, -1, head_size});
    const auto beam_idx = o.gathered_states ? param(element::i32, PartialShape{-1}) : nullptr;

    SinkVector sinks;
    std::vector<std::shared_ptr<op::util::Variable>> variables;
    const auto read_state = [&](const std::string& id, const PartialShape& shape) -> Output<Node> {
        if (!o.variable_states) {
            return param(element::f32, shape);
        }
        variables.push_back(std::make_shared<op::util::Variable>(op::util::VariableInfo{shape, element::f32, id}));
        Output<Node> state = std::make_shared<v6::ReadValue>(variables.back());
        if (o.gathered_states) {
            state = std::make_shared<v8::Gather>(state, beam_idx, v0::Constant::create(element::i64, Shape{}, {0}));
        }
        return state;
    };
    const auto past_block_keys = read_state("block_keys", PartialShape{-1, -1, head_size});
    const auto incomplete_raw_keys = read_state("incomplete_raw_keys", PartialShape{-1, key_block_size - 1, head_size});
    const auto incomplete_rope = read_state("incomplete_rope", PartialShape{-1, 2 * rotary_dim});
    const auto block_keys_cache = std::make_shared<v0::Concat>(
        OutputVector{std::make_shared<v0::Unsqueeze>(past_block_keys, i64({1})), cache_padding},
        2);
    const auto incomplete_raw_keys_with_head = std::make_shared<v0::Unsqueeze>(incomplete_raw_keys, i64({1}));
    const auto q_norm_weight = v0::Constant::create(element::f32, Shape{head_size}, {1.0f});
    const auto k_norm_weight = v0::Constant::create(element::f32, Shape{head_size}, {0.5f});

    op_internal::QSAIndexer::Config config;
    config.key_block_size = key_block_size;
    config.block_topk = block_topk;
    config.num_selection_heads = selection_heads;
    config.indexer_rotary_dim = rotary_dim;
    config.indexer_head_dim = head_size;
    config.epsilon = epsilon;
    config.query_norm = o.query_norm_rope;
    config.query_rope = o.query_norm_rope;

    Output<Node> indices, count, block_keys_state, raw_keys_state, rope_state;
    if (paged) {
        const auto past_lens = make_named_param("past_lens", element::i32, PartialShape{-1});
        const auto subsequence_begins = make_named_param("subsequence_begins", element::i32, PartialShape{-1});
        const auto block_indices = make_named_param("block_indices", element::i32, PartialShape{-1});
        const auto block_indices_begins = make_named_param("block_indices_begins", element::i32, PartialShape{-1});
        const auto index_key_cache =
            make_named_param("index_key_cache.0", element::f32, PartialShape{-1, key_heads, -1, head_size});
        const auto index_block_rope_cache =
            make_named_param("index_block_rope_cache.0", element::f32, PartialShape{-1, -1, 2 * rotary_dim});
        const auto block_index_keys_cache =
            make_named_param("block_index_keys_cache.0", element::f32, PartialShape{-1, key_heads, -1, head_size});
        params.insert(params.end(),
                      {past_lens,
                       subsequence_begins,
                       block_indices,
                       block_indices_begins,
                       index_key_cache,
                       index_block_rope_cache,
                       block_index_keys_cache});

        const auto flatten = [](const Output<Node>& data, int64_t width) {
            return std::make_shared<v1::Reshape>(data, i64({-1, width}), false);
        };
        OutputVector inputs(PagedQSAIndexerInputIdx::COUNT);
        inputs[PagedQSAIndexerInputIdx::INDEX_QUERY] = flatten(query, query_heads * head_size);
        inputs[PagedQSAIndexerInputIdx::RAW_INDEX_KEYS] = flatten(raw_keys, key_heads * head_size);
        inputs[PagedQSAIndexerInputIdx::ROPE_COS] = flatten(rope_cos, rotary_dim);
        inputs[PagedQSAIndexerInputIdx::ROPE_SIN] = flatten(rope_sin, rotary_dim);
        inputs[PagedQSAIndexerInputIdx::Q_NORM_WEIGHT] = q_norm_weight;
        inputs[PagedQSAIndexerInputIdx::K_NORM_WEIGHT] = k_norm_weight;
        inputs[PagedQSAIndexerInputIdx::PAST_LENS] = past_lens;
        inputs[PagedQSAIndexerInputIdx::SUBSEQUENCE_BEGINS] = subsequence_begins;
        inputs[PagedQSAIndexerInputIdx::BLOCK_INDICES] = block_indices;
        inputs[PagedQSAIndexerInputIdx::BLOCK_INDICES_BEGINS] = block_indices_begins;
        inputs[PagedQSAIndexerInputIdx::INDEX_KEY_CACHE] = index_key_cache;
        inputs[PagedQSAIndexerInputIdx::INDEX_BLOCK_ROPE_CACHE] = index_block_rope_cache;
        inputs[PagedQSAIndexerInputIdx::BLOCK_INDEX_KEYS_CACHE] = block_index_keys_cache;
        op_internal::PagedQSAIndexer::Config paged_config;
        paged_config.key_block_size = config.key_block_size;
        paged_config.block_topk = config.block_topk;
        paged_config.num_selection_heads = config.num_selection_heads;
        paged_config.indexer_rotary_dim = config.indexer_rotary_dim;
        paged_config.indexer_head_dim = config.indexer_head_dim;
        paged_config.epsilon = config.epsilon;
        const auto paged_indexer = std::make_shared<op_internal::PagedQSAIndexer>(inputs, paged_config);

        const auto batch_and_length = op::util::node_to_get_shape_value_of_indices_from_shape_source(query, {0, 1});
        const auto restore = [&](size_t output, const std::vector<int64_t>& tail, const std::vector<int64_t>& order) {
            const auto shape = std::make_shared<v0::Concat>(OutputVector{batch_and_length, i64(tail)}, 0);
            const auto token_major = std::make_shared<v1::Reshape>(paged_indexer->output(output), shape, false);
            return std::make_shared<v1::Transpose>(token_major, i64(order));
        };
        indices =
            restore(PagedQSAIndexerOutputIdx::SELECTED_BLOCK_INDICES, {selection_heads, block_topk}, {0, 2, 1, 3});
        count = restore(PagedQSAIndexerOutputIdx::SELECTED_BLOCK_COUNT, {selection_heads}, {0, 2, 1});
        block_keys_state = block_keys_cache;
        raw_keys_state = incomplete_raw_keys_with_head;
        rope_state = incomplete_rope;
    } else {
        OutputVector inputs(QSAIndexerInputIdx::COUNT);
        inputs[QSAIndexerInputIdx::INDEX_QUERY] = query;
        inputs[QSAIndexerInputIdx::RAW_INDEX_KEYS] = raw_keys;
        inputs[QSAIndexerInputIdx::ROPE_COS] = rope_cos;
        inputs[QSAIndexerInputIdx::ROPE_SIN] = rope_sin;
        inputs[QSAIndexerInputIdx::KEY_LENGTH] = key_length;
        inputs[QSAIndexerInputIdx::Q_NORM_WEIGHT] = q_norm_weight;
        inputs[QSAIndexerInputIdx::K_NORM_WEIGHT] = k_norm_weight;
        inputs[QSAIndexerInputIdx::BLOCK_INDEX_KEYS_CACHE] = block_keys_cache;
        inputs[QSAIndexerInputIdx::PAST_INCOMPLETE_RAW_K] = incomplete_raw_keys_with_head;
        inputs[QSAIndexerInputIdx::PAST_INCOMPLETE_ROPE] = incomplete_rope;
        const auto indexer = std::make_shared<op_internal::QSAIndexer>(inputs, config);
        indices = indexer->output(QSAIndexerOutputIdx::SELECTED_BLOCK_INDICES);
        count = indexer->output(QSAIndexerOutputIdx::SELECTED_BLOCK_COUNT);
        block_keys_state = indexer->output(QSAIndexerOutputIdx::BLOCK_INDEX_KEYS_CACHE);
        raw_keys_state = indexer->output(QSAIndexerOutputIdx::PAST_INCOMPLETE_RAW_K);
        rope_state = indexer->output(QSAIndexerOutputIdx::PAST_INCOMPLETE_ROPE);
    }

    ResultVector results{std::make_shared<v0::Result>(indices), std::make_shared<v0::Result>(count)};
    const OutputVector states{std::make_shared<v0::Squeeze>(block_keys_state, i64({1})),
                              std::make_shared<v0::Squeeze>(raw_keys_state, i64({1})),
                              rope_state};
    for (size_t i = 0; i < states.size(); ++i) {
        if (o.variable_states) {
            sinks.push_back(std::make_shared<v6::Assign>(states[i], variables[i]));
        } else {
            results.push_back(std::make_shared<v0::Result>(states[i]));
        }
    }
    return std::make_shared<Model>(results, sinks, params);
}

std::unordered_set<std::string> run_paged_qsa_indexer_fusion(const std::shared_ptr<Model>& model) {
    pass::paged_attention::PaParams pa_params{model->get_parameters()};
    std::unordered_set<std::string> var_ids_to_remove;
    pass::Manager manager;
    manager.set_per_pass_validation(false);
    manager.register_pass<pass::PagedQSAIndexerFusion>(pa_params, var_ids_to_remove);
    manager.run_passes(model);
    model->add_parameters(pa_params.items());
    model->validate_nodes_and_infer_types();
    return var_ids_to_remove;
}

class PagedQSAIndexerFusionTest : public TransformationTestsF, public testing::WithParamInterface<GraphOptions> {};

class PagedQSAIndexerFusionConvertedTest : public PagedQSAIndexerFusionTest {};

TEST_P(PagedQSAIndexerFusionConvertedTest, ConvertsToPagedQSAIndexer) {
    model = make_model(GetParam(), false);
    model_ref = make_model(GetParam(), true);
    disable_rt_info_check();
    const auto var_ids_to_remove = run_paged_qsa_indexer_fusion(model);
    EXPECT_EQ(var_ids_to_remove,
              (std::unordered_set<std::string>{"block_keys", "incomplete_raw_keys", "incomplete_rope"}));
}

INSTANTIATE_TEST_SUITE_P(PagedQSAIndexerFusion,
                         PagedQSAIndexerFusionConvertedTest,
                         testing::Values(GraphOptions{}, GraphOptions{true, true}));

class PagedQSAIndexerFusionNotConvertedTest : public PagedQSAIndexerFusionTest {};

TEST_P(PagedQSAIndexerFusionNotConvertedTest, KeepsQSAIndexer) {
    model = make_model(GetParam(), false);
    model_ref = make_model(GetParam(), false);
    EXPECT_TRUE(run_paged_qsa_indexer_fusion(model).empty());
}

// Without query_norm and query_rope, and with states that are not read from variables.
INSTANTIATE_TEST_SUITE_P(PagedQSAIndexerFusion,
                         PagedQSAIndexerFusionNotConvertedTest,
                         testing::Values(GraphOptions{false}, GraphOptions{true, false, false}));

// A Qwen sparse attention layer: QSAPrepare and SparseAttentionIndexer select the blocks of SparseAttention.
std::shared_ptr<Model> make_sparse_attention_model(bool query_rope,
                                                   element::Type key_length_type = element::i32,
                                                   bool decomposed_query = false,
                                                   bool with_sdpa_layer = false) {
    const auto rotary = decomposed_query ? head_size : rotary_dim;
    const auto input_ids = make_named_param("input_ids", element::i64, PartialShape{-1, -1});
    const auto beam_idx = make_named_param("beam_idx", element::i32, PartialShape{-1});
    const auto index_query =
        std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, query_heads, head_size});
    const auto raw_keys = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, head_size});
    const auto rope_cos = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, rotary});
    const auto rope_sin = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, rotary});
    const auto key_length = std::make_shared<v0::Parameter>(key_length_type, PartialShape{-1});
    const auto query = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, query_heads, -1, head_size});
    const auto key = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, kv_heads, -1, head_size});
    const auto value = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, kv_heads, -1, head_size});
    ParameterVector
        params{input_ids, beam_idx, index_query, raw_keys, rope_cos, rope_sin, key_length, query, key, value};

    SinkVector sinks;
    const auto state = [&](const std::string& id, const PartialShape& shape, const Shape& init_shape) {
        const auto variable = std::make_shared<op::util::Variable>(op::util::VariableInfo{shape, element::f32, id});
        const auto init = v0::Constant::create(element::f32, init_shape, std::vector<float>(shape_size(init_shape)));
        const auto read_value = std::make_shared<v6::ReadValue>(init, variable);
        const auto gathered =
            std::make_shared<v8::Gather>(read_value, beam_idx, v0::Constant::create(element::i64, Shape{}, {0}));
        return std::make_pair(gathered, variable);
    };
    const auto assign = [&sinks](const Output<Node>& data, const std::shared_ptr<op::util::Variable>& variable) {
        sinks.push_back(std::make_shared<v6::Assign>(data, variable));
    };

    Output<Node> rms;
    if (decomposed_query) {
        const auto square = std::make_shared<v1::Power>(index_query, v0::Constant::create(element::f32, Shape{}, {2}));
        const auto mean = std::make_shared<v1::ReduceMean>(square, i64({-1}), true);
        const auto sum = std::make_shared<v1::Add>(mean, v0::Constant::create(element::f32, Shape{}, {epsilon}));
        const auto inverse = std::make_shared<v1::Divide>(v0::Constant::create(element::f32, Shape{}, {1}),
                                                          std::make_shared<v0::Sqrt>(sum));
        rms = std::make_shared<v1::Multiply>(index_query, inverse);
    } else {
        rms = std::make_shared<op_internal::RMS>(index_query, epsilon, element::f32);
    }
    Output<Node> indexer_query;
    if (decomposed_query) {
        const auto x = std::make_shared<v1::Transpose>(rms, i64({0, 2, 1, 3}));
        const auto last_axis = x->get_output_partial_shape(0).rank().get_length() - 1;
        const auto halves = std::make_shared<v1::VariadicSplit>(x, i64({last_axis}), i64({rotary / 2, rotary / 2}));
        const auto negative =
            std::make_shared<v1::Multiply>(halves->output(1), v0::Constant::create(element::f32, Shape{}, {-1}));
        const auto rotated = std::make_shared<v0::Concat>(OutputVector{negative, halves->output(0)}, -1);
        const auto mul_cos = std::make_shared<v1::Multiply>(x, std::make_shared<v0::Unsqueeze>(rope_cos, i64({1})));
        const auto mul_sin =
            std::make_shared<v1::Multiply>(rotated, std::make_shared<v0::Unsqueeze>(rope_sin, i64({1})));
        indexer_query = std::make_shared<v1::Add>(mul_cos, mul_sin);
    } else if (query_rope) {
        op_internal::RoPE::Config rope_config;
        rope_config.rotary_ndims = rotary;
        rope_config.input_trans0213 = true;
        indexer_query =
            std::make_shared<op_internal::RoPE>(OutputVector{rms,
                                                             std::make_shared<v0::Unsqueeze>(rope_cos, i64({1})),
                                                             std::make_shared<v0::Unsqueeze>(rope_sin, i64({1}))},
                                                rope_config);
    } else {
        indexer_query = std::make_shared<v1::Transpose>(rms, i64({0, 2, 1, 3}));
    }

    const auto [block_keys, block_keys_var] =
        state("block_keys", PartialShape{-1, -1, head_size}, Shape{1, 0, head_size});
    const auto [raw_k, raw_k_var] = state("incomplete_raw_keys",
                                          PartialShape{-1, key_block_size - 1, head_size},
                                          Shape{1, key_block_size - 1, head_size});
    const auto [rope_row, rope_row_var] =
        state("incomplete_rope", PartialShape{-1, 2 * rotary}, Shape{1, static_cast<size_t>(2 * rotary)});
    const auto prepare = std::make_shared<v17::QSAPrepare>(raw_keys,
                                                           block_keys,
                                                           key_length,
                                                           rope_cos,
                                                           rope_sin,
                                                           v0::Constant::create(element::f32, Shape{head_size}, {0.5f}),
                                                           raw_k,
                                                           rope_row,
                                                           key_block_size,
                                                           epsilon);
    const auto indexer = std::make_shared<v17::SparseAttentionIndexer>(indexer_query,
                                                                       prepare->output(0),
                                                                       key_length,
                                                                       key_block_size,
                                                                       block_topk * key_block_size,
                                                                       selection_heads);
    assign(prepare->output(0), block_keys_var);
    assign(prepare->output(1), raw_k_var);
    assign(prepare->output(2), rope_row_var);

    const auto [past_key, key_var] =
        state("past_key", PartialShape{-1, kv_heads, -1, head_size}, Shape{1, kv_heads, 0, head_size});
    const auto [past_value, value_var] =
        state("past_value", PartialShape{-1, kv_heads, -1, head_size}, Shape{1, kv_heads, 0, head_size});
    const auto full_key = std::make_shared<v0::Concat>(OutputVector{past_key, key}, 2);
    const auto full_value = std::make_shared<v0::Concat>(OutputVector{past_value, value}, 2);
    assign(full_key, key_var);
    assign(full_value, value_var);
    const auto attention = std::make_shared<v17::SparseAttention>(query,
                                                                  full_key,
                                                                  full_value,
                                                                  indexer->output(0),
                                                                  indexer->output(1),
                                                                  key_length,
                                                                  key_block_size);
    ResultVector results{std::make_shared<v0::Result>(attention)};
    if (with_sdpa_layer) {
        // A dense attention layer before the sparse one, as in models mixing both attention kinds.
        const auto sdpa_query =
            std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, kv_heads, -1, head_size});
        const auto sdpa_key = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, kv_heads, -1, head_size});
        const auto sdpa_value =
            std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, kv_heads, -1, head_size});
        const auto sdpa_mask = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, 1, -1, -1});
        params.insert(params.end(), {sdpa_query, sdpa_key, sdpa_value, sdpa_mask});
        const auto [sdpa_past_key, sdpa_key_var] =
            state("sdpa_past_key", PartialShape{-1, kv_heads, -1, head_size}, Shape{1, kv_heads, 0, head_size});
        const auto [sdpa_past_value, sdpa_value_var] =
            state("sdpa_past_value", PartialShape{-1, kv_heads, -1, head_size}, Shape{1, kv_heads, 0, head_size});
        const auto sdpa_full_key = std::make_shared<v0::Concat>(OutputVector{sdpa_past_key, sdpa_key}, 2);
        const auto sdpa_full_value = std::make_shared<v0::Concat>(OutputVector{sdpa_past_value, sdpa_value}, 2);
        assign(sdpa_full_key, sdpa_key_var);
        assign(sdpa_full_value, sdpa_value_var);
        const auto sdpa = std::make_shared<v13::ScaledDotProductAttention>(sdpa_query,
                                                                           sdpa_full_key,
                                                                           sdpa_full_value,
                                                                           sdpa_mask,
                                                                           false);
        results.push_back(std::make_shared<v0::Result>(sdpa));
    }
    return std::make_shared<Model>(results, sinks, params);
}

size_t count_ops(const std::shared_ptr<Model>& model, const DiscreteTypeInfo& type) {
    size_t count = 0;
    for (const auto& node : model->get_ops()) {
        count += node->get_type_info() == type ? 1 : 0;
    }
    return count;
}

class SDPAToPagedAttentionQSATest : public testing::Test, public testing::WithParamInterface<bool> {};

TEST_P(SDPAToPagedAttentionQSATest, ConvertsSparseAttention) {
    const auto model = make_sparse_attention_model(true, element::i64, GetParam());
    pass::Manager manager;
    manager.register_pass<pass::SDPAToPagedAttention>();
    manager.run_passes(model);

    EXPECT_EQ(count_ops(model, op_internal::PagedQSAIndexer::get_type_info_static()), 1);
    EXPECT_EQ(count_ops(model, op_internal::PagedSparseAttention::get_type_info_static()), 1);
    EXPECT_EQ(count_ops(model, op_internal::QSAIndexer::get_type_info_static()), 0);
    EXPECT_EQ(count_ops(model, v17::SparseAttention::get_type_info_static()), 0);
    EXPECT_EQ(count_ops(model, v17::SparseAttentionIndexer::get_type_info_static()), 0);
    EXPECT_EQ(count_ops(model, op_internal::RMS::get_type_info_static()), 0);
    EXPECT_EQ(count_ops(model, op_internal::RoPE::get_type_info_static()), 0);
    EXPECT_EQ(count_ops(model, v1::ReduceMean::get_type_info_static()), 0);
    EXPECT_EQ(count_ops(model, v1::VariadicSplit::get_type_info_static()), 0);
    EXPECT_TRUE(model->get_sinks().empty());
    for (const auto& name : {"key_cache.0",
                             "value_cache.0",
                             "index_key_cache.0",
                             "index_block_rope_cache.0",
                             "block_index_keys_cache.0",
                             "block_indices"}) {
        EXPECT_NO_THROW(model->input(name)) << name;
    }
}

INSTANTIATE_TEST_SUITE_P(PagedQSAIndexerFusion,
                         SDPAToPagedAttentionQSATest,
                         testing::Values(false, true),
                         [](const testing::TestParamInfo<bool>& info) {
                             return info.param ? "DecomposedQuery" : "FusedQuery";
                         });

TEST(PagedQSAIndexerFusion, SDPAToPagedAttentionKeepsCacheInputOrderWithSDPA) {
    const auto model = make_sparse_attention_model(true, element::i64, true, true);
    pass::Manager manager;
    manager.register_pass<pass::SDPAToPagedAttention>();
    manager.run_passes(model);

    EXPECT_EQ(count_ops(model, op_internal::PagedQSAIndexer::get_type_info_static()), 1);
    EXPECT_EQ(count_ops(model, op_internal::PagedSparseAttention::get_type_info_static()), 1);
    EXPECT_EQ(count_ops(model, v17::SparseAttention::get_type_info_static()), 0);
    EXPECT_EQ(count_ops(model, v13::ScaledDotProductAttention::get_type_info_static()), 0);
    EXPECT_TRUE(model->get_sinks().empty());

    // QSA index caches come first, then the SDPA layer caches, then the SparseAttention layer caches.
    std::vector<std::string> cache_names;
    for (const auto& input : model->inputs()) {
        for (const auto& name : input.get_names()) {
            if (name.find("cache.") != std::string::npos) {
                cache_names.push_back(name);
            }
        }
    }
    const std::vector<std::string> expected{"index_key_cache.0",
                                            "index_block_rope_cache.0",
                                            "block_index_keys_cache.0",
                                            "key_cache.0",
                                            "value_cache.0",
                                            "key_cache.1",
                                            "value_cache.1"};
    EXPECT_EQ(cache_names, expected);
}

struct DisabledFusion {
    DiscreteTypeInfo pass_type;
    DiscreteTypeInfo preserved_op_type;
};

class SDPAToPagedAttentionDisabledFusionTest : public testing::Test,
                                               public testing::WithParamInterface<DisabledFusion> {};

TEST_P(SDPAToPagedAttentionDisabledFusionTest, DoesNotBypassDisabledFusion) {
    const auto model = make_sparse_attention_model(true, element::i64, true);
    pass::Manager manager;
    manager.register_pass<pass::SDPAToPagedAttention>();
    manager.get_pass_config()->disable(GetParam().pass_type);
    EXPECT_THROW(manager.run_passes(model), ov::Exception);

    EXPECT_EQ(count_ops(model, GetParam().preserved_op_type), 1);
    EXPECT_EQ(count_ops(model, op_internal::PagedQSAIndexer::get_type_info_static()), 0);
    EXPECT_EQ(count_ops(model, op_internal::PagedSparseAttention::get_type_info_static()), 0);
}

INSTANTIATE_TEST_SUITE_P(
    PagedQSAIndexerFusion,
    SDPAToPagedAttentionDisabledFusionTest,
    testing::Values(
        DisabledFusion{pass::RoPEFusion::get_type_info_static(), v1::VariadicSplit::get_type_info_static()},
        DisabledFusion{pass::RoPEFusionGPTNEOX::get_type_info_static(), v1::VariadicSplit::get_type_info_static()},
        DisabledFusion{pass::RMSFusion::get_type_info_static(), v1::ReduceMean::get_type_info_static()},
        DisabledFusion{pass::RMSFusionMatcher::get_type_info_static(), v1::ReduceMean::get_type_info_static()},
        DisabledFusion{pass::FuseQSAIndexerQueryRoPE::get_type_info_static(),
                       op_internal::RoPE::get_type_info_static()},
        DisabledFusion{pass::FuseQSAIndexerQueryNorm::get_type_info_static(), op_internal::RMS::get_type_info_static()},
        DisabledFusion{pass::PagedQSAIndexerFusion::get_type_info_static(),
                       op_internal::QSAIndexer::get_type_info_static()}),
    [](const testing::TestParamInfo<DisabledFusion>& info) {
        return std::string(info.param.pass_type.name);
    });

TEST(PagedQSAIndexerFusion, SDPAToPagedAttentionHonorsRMSCallback) {
    const auto model = make_sparse_attention_model(true, element::i64, true);
    pass::Manager manager;
    manager.register_pass<pass::SDPAToPagedAttention>();
    size_t callback_count = 0;
    manager.get_pass_config()->set_callback<pass::RMSFusionMatcher>([&](const std::shared_ptr<const Node>&) {
        ++callback_count;
        return true;
    });
    EXPECT_THROW(manager.run_passes(model), ov::Exception);

    EXPECT_GT(callback_count, 0);
    EXPECT_EQ(count_ops(model, v1::ReduceMean::get_type_info_static()), 1);
    EXPECT_EQ(count_ops(model, op_internal::RMS::get_type_info_static()), 0);
    EXPECT_EQ(count_ops(model, op_internal::PagedQSAIndexer::get_type_info_static()), 0);
    EXPECT_EQ(count_ops(model, op_internal::PagedSparseAttention::get_type_info_static()), 0);
}

TEST(PagedQSAIndexerFusion, SDPAToPagedAttentionRejectsUnconvertedSparseAttention) {
    const auto model = make_sparse_attention_model(false);
    pass::Manager manager;
    manager.register_pass<pass::SDPAToPagedAttention>();
    EXPECT_THROW(manager.run_passes(model), ov::Exception);
}

}  // namespace
