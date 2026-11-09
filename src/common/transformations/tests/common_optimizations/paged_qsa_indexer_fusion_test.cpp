// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/paged_qsa_indexer_fusion.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
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
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sparse_attention.hpp"
#include "openvino/op/sparse_attention_indexer.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"
#include "ov_ops/rms.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "transformations/common_optimizations/fuse_qsa_indexer.hpp"
#include "transformations/common_optimizations/fuse_rotary_positional_embeddings.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"
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
    bool query_norm = true;
    bool gathered_states = false;
    bool variable_states = true;
    bool query_rope = true;
    int64_t rope_length = -1;
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
    const auto rope_cos = param(element::f32, PartialShape{-1, o.rope_length, rotary_dim});
    const auto rope_sin = param(element::f32, PartialShape{-1, o.rope_length, rotary_dim});
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
    config.query_norm = o.query_norm;
    config.query_rope = o.query_rope;

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
        paged_config.query_norm = config.query_norm;
        paged_config.query_rope = config.query_rope;
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
    comparator.enable(FunctionsComparator::ATTRIBUTES);
    disable_rt_info_check();
    const auto var_ids_to_remove = run_paged_qsa_indexer_fusion(model);
    EXPECT_EQ(var_ids_to_remove,
              (std::unordered_set<std::string>{"block_keys", "incomplete_raw_keys", "incomplete_rope"}));
}

INSTANTIATE_TEST_SUITE_P(PagedQSAIndexerFusion,
                         PagedQSAIndexerFusionConvertedTest,
                         testing::Values(GraphOptions{},
                                         GraphOptions{true, true},
                                         GraphOptions{false},
                                         GraphOptions{true, false, true, false},
                                         GraphOptions{false, false, true, false}));

class PagedQSAIndexerFusionNotConvertedTest : public PagedQSAIndexerFusionTest {};

TEST_P(PagedQSAIndexerFusionNotConvertedTest, KeepsQSAIndexer) {
    model = make_model(GetParam(), false);
    model_ref = make_model(GetParam(), false);
    EXPECT_TRUE(run_paged_qsa_indexer_fusion(model).empty());
}

// States not read from variables cannot be converted to paged caches.
INSTANTIATE_TEST_SUITE_P(PagedQSAIndexerFusion,
                         PagedQSAIndexerFusionNotConvertedTest,
                         testing::Values(GraphOptions{true, false, false}));

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

struct QueryOptions {
    std::string name;
    float norm_epsilon = epsilon;
    int64_t norm_axis = -1;
    bool affine = false;
    bool changes_precision = false;
    bool distinct_cos = false;
    bool distinct_sin = false;
    bool interleaved = false;
    bool extra_consumers = false;
    bool with_rope = true;
    bool with_norm = true;
    bool flat_norm = false;
    bool non_unit_unused_weight = false;
    bool rope_before_norm = false;
    bool indexer_query_norm = false;
    bool indexer_query_rope = false;
    bool absorb_norm = true;
    bool absorb_rope = true;
    bool rope_rows_special_zero = false;
    int64_t rope_length = -1;
    int64_t rope_axis = 1;
    // Reshape target expressing the cos / sin head-axis Unsqueeze; empty keeps the Unsqueeze.
    std::vector<int64_t> rope_rows_target;
};

template <class Indexer = op_internal::PagedQSAIndexer>
std::shared_ptr<Model> make_query_model(const QueryOptions& o, bool fused) {
    using In = typename Indexer::InputIdx;
    constexpr bool paged = std::is_same<Indexer, op_internal::PagedQSAIndexer>::value;
    const auto model = make_model(GraphOptions{false, false, paged, false, o.rope_length}, paged);
    for (const auto& node : model->get_ops()) {
        const auto indexer = ov::as_type_ptr<Indexer>(node);
        if (!indexer) {
            continue;
        }
        auto inputs = indexer->input_values();
        if (o.non_unit_unused_weight && !(fused && o.absorb_norm)) {
            inputs[In::Q_NORM_WEIGHT] = v0::Constant::create(element::f32, Shape{head_size}, {0.5f});
        }
        const auto unflatten = [paged](const Output<Node>& input) {
            if constexpr (paged) {
                return input.get_node_shared_ptr()->input_value(0);
            } else {
                return input;
            }
        };
        Output<Node> raw_query = unflatten(inputs[In::INDEX_QUERY]);
        if (o.flat_norm) {
            raw_query = inputs[In::INDEX_QUERY];
        }
        const auto make_norm = [&](const Output<Node>& input) -> Output<Node> {
            const auto norm_input =
                o.changes_precision ? Output<Node>(std::make_shared<v0::Convert>(input, element::f16)) : input;
            if (o.affine) {
                return std::make_shared<op_internal::RMS>(norm_input,
                                                          v0::Constant::create(element::f32, Shape{head_size}, {0.5f}),
                                                          o.norm_epsilon,
                                                          element::f32,
                                                          o.norm_axis);
            }
            return std::make_shared<op_internal::RMS>(norm_input, o.norm_epsilon, element::f32, o.norm_axis);
        };
        const auto other_rows = [&](const Output<Node>& rows) -> Output<Node> {
            const auto param = std::make_shared<v0::Parameter>(rows.get_element_type(), rows.get_partial_shape());
            model->add_parameters({param});
            return param;
        };
        const auto cos = unflatten(inputs[In::ROPE_COS]);
        const auto sin = unflatten(inputs[In::ROPE_SIN]);
        std::shared_ptr<Node> rope;
        const auto make_rope = [&](const Output<Node>& input) -> Output<Node> {
            op_internal::RoPE::Config config;
            config.rotary_ndims = rotary_dim;
            config.input_trans0213 = true;
            config.is_interleaved = o.interleaved;
            const auto with_head_axis = [&o](const Output<Node>& rows) -> Output<Node> {
                if (o.rope_rows_target.empty()) {
                    return std::make_shared<v0::Unsqueeze>(rows, i64({o.rope_axis}));
                }
                return std::make_shared<v1::Reshape>(rows, i64(o.rope_rows_target), o.rope_rows_special_zero);
            };
            rope = std::make_shared<op_internal::RoPE>(
                OutputVector{input,
                             with_head_axis(o.distinct_cos ? other_rows(cos) : cos),
                             with_head_axis(o.distinct_sin ? other_rows(sin) : sin)},
                config);
            return std::make_shared<v1::Transpose>(rope, i64({0, 2, 1, 3}));
        };
        Output<Node> norm = raw_query;
        Output<Node> rotated;
        Output<Node> query = raw_query;
        if (o.rope_before_norm) {
            rotated = make_rope(raw_query);
            if (o.with_norm) {
                query = norm = make_norm(rotated);
            } else {
                query = rotated;
            }
        } else {
            if (o.with_norm) {
                query = norm = make_norm(raw_query);
            }
            if (o.with_rope) {
                query = rotated = make_rope(query);
            }
        }
        if (o.extra_consumers) {
            model->add_results({std::make_shared<v0::Result>(rope), std::make_shared<v0::Result>(norm)});
        }
        auto config = indexer->get_config();
        config.query_norm = o.indexer_query_norm;
        config.query_rope = o.indexer_query_rope;
        if (fused) {
            config.query_norm = o.indexer_query_norm || (o.with_norm && o.absorb_norm);
            config.query_rope = o.indexer_query_rope || (o.with_rope && o.absorb_rope);
            const bool norm_outside = o.with_norm && !o.absorb_norm;
            const bool rope_outside = o.with_rope && !o.absorb_rope;
            if (!norm_outside && !rope_outside) {
                query = raw_query;
            } else if (!rope_outside) {
                query = norm;
            } else if (!norm_outside) {
                query = rotated;
            }
        }
        if constexpr (paged) {
            const auto flatten = inputs[In::INDEX_QUERY].get_node_shared_ptr();
            inputs[In::INDEX_QUERY] =
                o.flat_norm ? query
                            : Output<Node>(std::make_shared<v1::Reshape>(query, flatten->input_value(1), false));
        } else {
            inputs[In::INDEX_QUERY] = query;
        }
        ov::replace_node(indexer, std::make_shared<Indexer>(inputs, config));
        break;
    }
    return model;
}

class FusePagedQSAIndexerQueryTest : public TransformationTestsF, public testing::WithParamInterface<QueryOptions> {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        comparator.enable(FunctionsComparator::ATTRIBUTES);
        comparator.enable(FunctionsComparator::CONST_VALUES);
        manager.register_pass<pass::FuseQSAIndexerQuery>();
    }
};

TEST_P(FusePagedQSAIndexerQueryTest, AbsorbsOnlyCompatibleQuerySteps) {
    model = make_query_model(GetParam(), false);
    model_ref = make_query_model(GetParam(), true);
}

const std::vector<QueryOptions> query_cases = [] {
    std::vector<QueryOptions> cases{QueryOptions{"partial_rotary"}};
    auto o = QueryOptions{"extra_consumers"};
    o.extra_consumers = true;
    cases.push_back(o);
    o = QueryOptions{"unused_query_weight"};
    o.non_unit_unused_weight = true;
    cases.push_back(o);
    o = QueryOptions{"without_rope"};
    o.with_rope = false;
    o.absorb_rope = false;
    cases.push_back(o);
    o = QueryOptions{"wrong_rope_axis"};
    o.rope_axis = 2;
    o.absorb_norm = o.absorb_rope = false;
    cases.push_back(o);
    o = QueryOptions{"flat_norm"};
    o.with_rope = false;
    o.flat_norm = true;
    o.absorb_norm = o.absorb_rope = false;
    cases.push_back(o);
    for (const auto& name : {"wrong_epsilon", "wrong_axis", "affine", "changes_precision"}) {
        o = QueryOptions{name};
        o.absorb_norm = false;
        if (o.name == "wrong_epsilon")
            o.norm_epsilon = 2 * epsilon;
        else if (o.name == "wrong_axis")
            o.norm_axis = -2;
        else if (o.name == "affine")
            o.affine = true;
        else
            o.changes_precision = true;
        cases.push_back(o);
    }
    // The RMS stage runs after the RoPE stage, so a RoPE after the RMS stays outside.
    o = QueryOptions{"rope_before_norm"};
    o.rope_before_norm = true;
    o.absorb_rope = false;
    cases.push_back(o);
    for (const auto& name : {"distinct_cos", "distinct_sin", "interleaved"}) {
        o = QueryOptions{name};
        o.absorb_norm = o.absorb_rope = false;
        o.distinct_cos = o.name == "distinct_cos";
        o.distinct_sin = o.name == "distinct_sin";
        o.interleaved = o.name == "interleaved";
        cases.push_back(o);
    }
    // NopElimination replaces the tiny model's head-axis Unsqueeze with this Reshape.
    o = QueryOptions{"tiny_model_rope_rows"};
    o.rope_length = 1;
    o.rope_rows_target = {-1, 1, 1, rotary_dim};
    cases.push_back(o);
    o = QueryOptions{"copy_rope_rows_dimension"};
    o.rope_length = 1;
    o.rope_rows_target = {0, 1, 1, rotary_dim};
    o.rope_rows_special_zero = true;
    cases.push_back(o);
    o = QueryOptions{"multiple_rope_rows"};
    o.rope_length = 2;
    o.rope_rows_target = {-1, 1, 1, rotary_dim};
    o.absorb_norm = o.absorb_rope = false;
    cases.push_back(o);
    o = QueryOptions{"moved_rope_rows"};
    o.rope_length = 1;
    o.rope_rows_target = {1, -1, 1, rotary_dim};
    o.absorb_norm = o.absorb_rope = false;
    cases.push_back(o);
    return cases;
}();

INSTANTIATE_TEST_SUITE_P(PagedQSAIndexerFusion,
                         FusePagedQSAIndexerQueryTest,
                         testing::ValuesIn(query_cases),
                         [](const testing::TestParamInfo<QueryOptions>& info) {
                             return info.param.name;
                         });

TEST_F(TransformationTestsF, FuseQSAIndexerQueryAfterNopElimination) {
    QueryOptions options;
    options.rope_length = 1;
    model = make_query_model(options, false);
    model_ref = make_query_model(options, true);

    pass::NopElimination nop_elimination;
    ASSERT_TRUE(nop_elimination.run_on_model(model));
    nop_elimination.run_on_model(model_ref);

    size_t rope_count = 0;
    for (const auto& node : model->get_ops()) {
        const auto rope = ov::as_type_ptr<op_internal::RoPE>(node);
        if (!rope) {
            continue;
        }
        ++rope_count;
        for (const auto idx : {1, 2}) {
            const auto reshape = ov::as_type_ptr<v1::Reshape>(rope->get_input_node_shared_ptr(idx));
            ASSERT_NE(reshape, nullptr);
            EXPECT_EQ(reshape->get_input_partial_shape(0), (PartialShape{-1, 1, rotary_dim}));
            EXPECT_FALSE(reshape->get_special_zero());
            const auto target = ov::as_type_ptr<v0::Constant>(reshape->get_input_node_shared_ptr(1));
            ASSERT_NE(target, nullptr);
            EXPECT_EQ(target->cast_vector<int64_t>(), (std::vector<int64_t>{-1, 1, 1, rotary_dim}));
        }
    }
    ASSERT_EQ(rope_count, 1);

    comparator.enable(FunctionsComparator::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CONST_VALUES);
    manager.register_pass<pass::FuseQSAIndexerQuery>();
}

TEST_F(TransformationTestsF, FuseQSAIndexerQueryRejectsDynamicRoPERowsTarget) {
    QueryOptions options;
    options.rope_length = 1;
    options.rope_rows_target = {-1, 1, 1, rotary_dim};
    model = make_query_model(options, false);

    const auto other = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1});
    const auto target =
        std::make_shared<v0::Concat>(OutputVector{std::make_shared<v0::ShapeOf>(other), i64({1, 1, rotary_dim})}, 0);
    model->add_parameters({other});
    size_t rope_count = 0;
    for (const auto& node : model->get_ops()) {
        const auto rope = ov::as_type_ptr<op_internal::RoPE>(node);
        if (!rope) {
            continue;
        }
        ++rope_count;
        for (const auto idx : {1, 2}) {
            const auto reshape = ov::as_type_ptr<v1::Reshape>(rope->get_input_node_shared_ptr(idx));
            ASSERT_NE(reshape, nullptr);
            reshape->input(1).replace_source_output(target);
            reshape->validate_and_infer_types();
            EXPECT_EQ(reshape->get_output_partial_shape(0), (PartialShape{-1, 1, 1, rotary_dim}));
        }
    }
    ASSERT_EQ(rope_count, 1);
    model->validate_nodes_and_infer_types();

    comparator.enable(FunctionsComparator::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CONST_VALUES);
    manager.register_pass<pass::FuseQSAIndexerQuery>();
}

std::shared_ptr<Model> make_mixed_query_model(const QueryOptions& options, bool dense_fused, bool paged_fused) {
    const auto mixed = make_query_model(options, paged_fused);
    const auto dense = make_query_model<op_internal::QSAIndexer>(options, dense_fused);
    mixed->add_parameters(dense->get_parameters());
    mixed->add_results(dense->get_results());
    return mixed;
}

TEST_F(TransformationTestsF, FuseQSAIndexerQueryHandlesMixedRoPERowsAdapters) {
    QueryOptions options;
    options.rope_length = 1;
    model = make_mixed_query_model(options, false, false);
    model_ref = make_mixed_query_model(options, true, true);

    size_t rope_count = 0;
    for (const auto& node : model->get_ops()) {
        const auto rope = ov::as_type_ptr<op_internal::RoPE>(node);
        if (!rope) {
            continue;
        }
        const auto idx = rope_count++ == 0 ? 1 : 2;
        const auto unsqueeze = ov::as_type_ptr<v0::Unsqueeze>(rope->get_input_node_shared_ptr(idx));
        ASSERT_NE(unsqueeze, nullptr);
        rope->input(idx).replace_source_output(
            std::make_shared<v1::Reshape>(unsqueeze->input_value(0), i64({-1, 1, 1, rotary_dim}), false));
        rope->validate_and_infer_types();
    }
    ASSERT_EQ(rope_count, 2);

    comparator.enable(FunctionsComparator::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CONST_VALUES);
    manager.register_pass<pass::FuseQSAIndexerQuery>();
}

TEST_F(TransformationTestsF, FuseQSAIndexerQueryHandlesDenseAndPagedIndexersIdempotently) {
    model = make_mixed_query_model(QueryOptions{}, false, false);
    model_ref = make_mixed_query_model(QueryOptions{}, true, true);
    comparator.enable(FunctionsComparator::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CONST_VALUES);
    manager.register_pass<pass::FuseQSAIndexerQuery>();
    manager.register_pass<pass::FuseQSAIndexerQuery>();
}

TEST_F(TransformationTestsF, FuseQSAIndexerQueryIgnoresInactiveNormWeights) {
    QueryOptions options;
    options.non_unit_unused_weight = true;
    model = make_mixed_query_model(options, false, false);
    model_ref = make_mixed_query_model(options, true, true);
    comparator.enable(FunctionsComparator::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CONST_VALUES);
    manager.register_pass<pass::FuseQSAIndexerQuery>();
}

TEST_F(TransformationTestsF, FuseQSAIndexerQueryNormIgnoresEnabledQueryRoPE) {
    QueryOptions options;
    options.with_rope = false;
    options.absorb_rope = false;
    options.indexer_query_rope = true;
    model = make_mixed_query_model(options, false, false);
    model_ref = make_mixed_query_model(options, true, true);
    comparator.enable(FunctionsComparator::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CONST_VALUES);
    manager.register_pass<pass::FuseQSAIndexerQuery>();
}

TEST_F(TransformationTestsF, FuseQSAIndexerQueryRoPEIgnoresEnabledQueryNorm) {
    QueryOptions options;
    options.with_norm = false;
    options.absorb_norm = false;
    options.indexer_query_norm = true;
    model = make_mixed_query_model(options, false, false);
    model_ref = make_mixed_query_model(options, true, true);
    comparator.enable(FunctionsComparator::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CONST_VALUES);
    manager.register_pass<pass::FuseQSAIndexerQuery>();
}

TEST_F(TransformationTestsF, FuseQSAIndexerQueryKeepsSharedQueryReshape) {
    const auto share_query_reshape = [](const std::shared_ptr<Model>& m) {
        for (const auto& node : m->get_ops()) {
            if (const auto indexer = ov::as_type_ptr<op_internal::PagedQSAIndexer>(node)) {
                m->add_results({std::make_shared<v0::Result>(
                    indexer->input_value(op_internal::PagedQSAIndexer::InputIdx::INDEX_QUERY))});
            }
        }
        return m;
    };
    model = share_query_reshape(make_query_model(QueryOptions{}, false));
    model_ref = share_query_reshape(make_query_model(QueryOptions{}, false));
    comparator.enable(FunctionsComparator::ATTRIBUTES);
    manager.register_pass<pass::FuseQSAIndexerQuery>();
}

TEST(FuseQSAIndexerQuery, UsesTypedRootsAndOptionalLayoutAdapters) {
    const auto verify = [](const std::shared_ptr<pass::MatcherPass>& matcher,
                           const DiscreteTypeInfo& expected_type,
                           const std::vector<size_t>& adapter_inputs) {
        const auto root = ov::as_type_ptr<pass::pattern::op::WrapType>(
            matcher->get_matcher()->get_pattern_value().get_node_shared_ptr());
        ASSERT_NE(root, nullptr);
        EXPECT_EQ(root->get_wrapped_types(), (std::vector<DiscreteTypeInfo>{expected_type}));
        for (const auto input : adapter_inputs) {
            const auto adapter =
                ov::as_type_ptr<pass::pattern::op::Optional>(root->input_value(input).get_node_shared_ptr());
            ASSERT_NE(adapter, nullptr);
            EXPECT_EQ(adapter->get_optional_types(),
                      (std::vector<DiscreteTypeInfo>{v1::Reshape::get_type_info_static()}));
        }
    };
    const auto& dense_type = op_internal::QSAIndexer::get_type_info_static();
    const auto& paged_type = op_internal::PagedQSAIndexer::get_type_info_static();
    verify(std::make_shared<pass::FuseQSAIndexerQueryRoPE<op_internal::QSAIndexer>>(),
           dense_type,
           {QSAIndexerInputIdx::INDEX_QUERY, QSAIndexerInputIdx::ROPE_COS, QSAIndexerInputIdx::ROPE_SIN});
    verify(std::make_shared<pass::FuseQSAIndexerQueryNorm<op_internal::QSAIndexer>>(),
           dense_type,
           {QSAIndexerInputIdx::INDEX_QUERY});
    verify(
        std::make_shared<pass::FuseQSAIndexerQueryRoPE<op_internal::PagedQSAIndexer>>(),
        paged_type,
        {PagedQSAIndexerInputIdx::INDEX_QUERY, PagedQSAIndexerInputIdx::ROPE_COS, PagedQSAIndexerInputIdx::ROPE_SIN});
    verify(std::make_shared<pass::FuseQSAIndexerQueryNorm<op_internal::PagedQSAIndexer>>(),
           paged_type,
           {PagedQSAIndexerInputIdx::INDEX_QUERY});
}

TEST(FuseQSAIndexerQuery, SpecializationsSharePassConfigTypes) {
    using DenseRoPE = pass::FuseQSAIndexerQueryRoPE<op_internal::QSAIndexer>;
    using PagedRoPE = pass::FuseQSAIndexerQueryRoPE<op_internal::PagedQSAIndexer>;
    using DenseNorm = pass::FuseQSAIndexerQueryNorm<op_internal::QSAIndexer>;
    using PagedNorm = pass::FuseQSAIndexerQueryNorm<op_internal::PagedQSAIndexer>;
    EXPECT_EQ(DenseRoPE::get_type_info_static(), PagedRoPE::get_type_info_static());
    EXPECT_EQ(DenseNorm::get_type_info_static(), PagedNorm::get_type_info_static());
}

template <class Broadcast = op::v3::Broadcast>
void broadcast_query_rope_rows(const std::shared_ptr<Model>& query_model, const std::vector<int64_t>& target_shape) {
    query_model->get_parameters().at(0)->set_partial_shape({2, 2, query_heads, head_size});
    std::shared_ptr<op_internal::PagedQSAIndexer> indexer;
    for (const auto& node : query_model->get_ops()) {
        indexer = ov::as_type_ptr<op_internal::PagedQSAIndexer>(node);
        if (indexer) {
            break;
        }
    }
    ASSERT_NE(indexer, nullptr);
    for (const auto idx : {PagedQSAIndexerInputIdx::ROPE_COS, PagedQSAIndexerInputIdx::ROPE_SIN}) {
        const auto reshape = ov::as_type_ptr<v1::Reshape>(indexer->input_value(idx).get_node_shared_ptr());
        ASSERT_NE(reshape, nullptr);
        const auto rows = ov::as_type_ptr<v0::Parameter>(reshape->input_value(0).get_node_shared_ptr());
        ASSERT_NE(rows, nullptr);
        rows->set_partial_shape({2, 1, rotary_dim});
        reshape->input(0).replace_source_output(std::make_shared<Broadcast>(rows, i64(target_shape)));
    }
    query_model->validate_nodes_and_infer_types();
}

TEST_F(TransformationTestsF, FuseQSAIndexerQueryRoPEAcceptsBatchBroadcast) {
    model = make_query_model(QueryOptions{}, false);
    model_ref = make_query_model(QueryOptions{}, true);
    broadcast_query_rope_rows(model, {2, 2, rotary_dim});
    broadcast_query_rope_rows(model_ref, {2, 2, rotary_dim});
    comparator.enable(FunctionsComparator::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CONST_VALUES);
    manager.register_pass<pass::FuseQSAIndexerQuery>();
}

// The CPU pipeline runs ConvertBroadcast3 before the query fusion.
TEST_F(TransformationTestsF, FuseQSAIndexerQueryRoPEAcceptsOpset1BatchBroadcast) {
    model = make_query_model(QueryOptions{}, false);
    model_ref = make_query_model(QueryOptions{}, true);
    broadcast_query_rope_rows<v1::Broadcast>(model, {2, 2, rotary_dim});
    broadcast_query_rope_rows<v1::Broadcast>(model_ref, {2, 2, rotary_dim});
    comparator.enable(FunctionsComparator::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CONST_VALUES);
    manager.register_pass<pass::FuseQSAIndexerQuery>();
}

class FuseQSAIndexerQueryModeTest : public TransformationTestsF, public testing::WithParamInterface<bool> {};

TEST_P(FuseQSAIndexerQueryModeTest, FusesOnlySelectedIndexerType) {
    model = make_mixed_query_model(QueryOptions{}, false, false);
    model_ref = make_mixed_query_model(QueryOptions{}, !GetParam(), GetParam());
    comparator.enable(FunctionsComparator::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CONST_VALUES);
    if (GetParam()) {
        manager.register_pass<pass::FuseQSAIndexerQueryRoPE<op_internal::PagedQSAIndexer>>();
        manager.register_pass<pass::FuseQSAIndexerQueryNorm<op_internal::PagedQSAIndexer>>();
    } else {
        manager.register_pass<pass::FuseQSAIndexerQueryRoPE<op_internal::QSAIndexer>>();
        manager.register_pass<pass::FuseQSAIndexerQueryNorm<op_internal::QSAIndexer>>();
    }
}

INSTANTIATE_TEST_SUITE_P(FuseQSAIndexer,
                         FuseQSAIndexerQueryModeTest,
                         testing::Values(false, true),
                         [](const testing::TestParamInfo<bool>& info) {
                             return info.param ? "Paged" : "Dense";
                         });

struct QueryPolicy {
    std::string name;
    bool paged;
    bool norm;
    bool callback;
};

class FuseQSAIndexerQueryPolicyTest : public TransformationTestsF, public testing::WithParamInterface<QueryPolicy> {};

TEST_P(FuseQSAIndexerQueryPolicyTest, HonorsIndividualMatcherConfiguration) {
    const auto& policy = GetParam();
    QueryOptions options;
    options.with_rope = !policy.norm;
    options.affine = !policy.norm;
    options.absorb_norm = policy.norm;
    options.absorb_rope = !policy.norm;
    model = make_mixed_query_model(options, false, false);
    model_ref = make_mixed_query_model(options, policy.callback && policy.paged, policy.callback && !policy.paged);
    comparator.enable(FunctionsComparator::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CONST_VALUES);
    manager.register_pass<pass::FuseQSAIndexerQuery>();
    const auto configure = [&](auto matcher) {
        using Pass = decltype(matcher);
        if (policy.callback) {
            manager.get_pass_config()->set_callback<Pass>(
                [paged = policy.paged](const std::shared_ptr<const Node>& node) {
                    return paged == ov::is_type<op_internal::PagedQSAIndexer>(node);
                });
        } else {
            manager.get_pass_config()->disable<Pass>();
        }
    };
    if (policy.norm && policy.paged)
        configure(pass::FuseQSAIndexerQueryNorm<op_internal::PagedQSAIndexer>{});
    else if (policy.norm)
        configure(pass::FuseQSAIndexerQueryNorm<op_internal::QSAIndexer>{});
    else if (policy.paged)
        configure(pass::FuseQSAIndexerQueryRoPE<op_internal::PagedQSAIndexer>{});
    else
        configure(pass::FuseQSAIndexerQueryRoPE<op_internal::QSAIndexer>{});
}

INSTANTIATE_TEST_SUITE_P(PagedQSAIndexerFusion,
                         FuseQSAIndexerQueryPolicyTest,
                         testing::Values(QueryPolicy{"DisableDenseRoPE", false, false, false},
                                         QueryPolicy{"DisableDenseNorm", false, true, false},
                                         QueryPolicy{"DisablePagedRoPE", true, false, false},
                                         QueryPolicy{"DisablePagedNorm", true, true, false},
                                         QueryPolicy{"DenseRoPECallback", false, false, true},
                                         QueryPolicy{"DenseNormCallback", false, true, true},
                                         QueryPolicy{"PagedRoPECallback", true, false, true},
                                         QueryPolicy{"PagedNormCallback", true, true, true}),
                         [](const testing::TestParamInfo<QueryPolicy>& info) {
                             return info.param.name;
                         });

void run_query_compilation_passes(const std::shared_ptr<Model>& model,
                                  const std::shared_ptr<pass::PassConfig>& config) {
    pass::Manager manager(config);
    manager.register_pass<pass::RMSFusion>(false, true);
    manager.register_pass<pass::RoPEFusion>(true);
    manager.register_pass<pass::FuseQSAIndexerQuery>();
    manager.run_passes(model);
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
    EXPECT_EQ(count_ops(model, op_internal::RMS::get_type_info_static()), GetParam() ? 0 : 1);
    EXPECT_EQ(count_ops(model, op_internal::RoPE::get_type_info_static()), GetParam() ? 0 : 1);
    EXPECT_EQ(count_ops(model, v1::ReduceMean::get_type_info_static()), GetParam() ? 1 : 0);
    EXPECT_EQ(count_ops(model, v1::VariadicSplit::get_type_info_static()), GetParam() ? 1 : 0);
    for (const auto& node : model->get_ops()) {
        if (const auto indexer = ov::as_type_ptr<op_internal::PagedQSAIndexer>(node)) {
            EXPECT_FALSE(indexer->get_config().query_norm);
            EXPECT_FALSE(indexer->get_config().query_rope);
        }
    }
    EXPECT_TRUE(model->get_sinks().empty());
    for (const auto& name : {"key_cache.0",
                             "value_cache.0",
                             "index_key_cache.0",
                             "index_block_rope_cache.0",
                             "block_index_keys_cache.0",
                             "block_indices"}) {
        EXPECT_NO_THROW(model->input(name)) << name;
    }
    run_query_compilation_passes(model, manager.get_pass_config());
    EXPECT_EQ(count_ops(model, op_internal::RMS::get_type_info_static()), 0);
    EXPECT_EQ(count_ops(model, op_internal::RoPE::get_type_info_static()), 0);
    EXPECT_EQ(count_ops(model, v1::ReduceMean::get_type_info_static()), 0);
    EXPECT_EQ(count_ops(model, v1::VariadicSplit::get_type_info_static()), 0);
    for (const auto& node : model->get_ops()) {
        if (const auto indexer = ov::as_type_ptr<op_internal::PagedQSAIndexer>(node)) {
            EXPECT_TRUE(indexer->get_config().query_norm);
            EXPECT_TRUE(indexer->get_config().query_rope);
        }
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
    bool prevents_conversion = false;
};

class SDPAToPagedAttentionDisabledFusionTest : public testing::Test,
                                               public testing::WithParamInterface<DisabledFusion> {};

TEST_P(SDPAToPagedAttentionDisabledFusionTest, DoesNotBypassDisabledFusion) {
    const auto model = make_sparse_attention_model(true, element::i64, true);
    pass::Manager manager;
    manager.register_pass<pass::SDPAToPagedAttention>();
    manager.get_pass_config()->disable(GetParam().pass_type);
    if (GetParam().prevents_conversion) {
        EXPECT_THROW(manager.run_passes(model), ov::Exception);
        EXPECT_EQ(count_ops(model, op_internal::PagedQSAIndexer::get_type_info_static()), 0);
        EXPECT_EQ(count_ops(model, op_internal::PagedSparseAttention::get_type_info_static()), 0);
    } else {
        EXPECT_NO_THROW(manager.run_passes(model));
        run_query_compilation_passes(model, manager.get_pass_config());
        EXPECT_EQ(count_ops(model, op_internal::PagedQSAIndexer::get_type_info_static()), 1);
        EXPECT_EQ(count_ops(model, op_internal::PagedSparseAttention::get_type_info_static()), 1);
    }
    EXPECT_EQ(count_ops(model, GetParam().preserved_op_type), 1);
}

INSTANTIATE_TEST_SUITE_P(
    PagedQSAIndexerFusion,
    SDPAToPagedAttentionDisabledFusionTest,
    testing::Values(
        DisabledFusion{pass::RoPEFusion::get_type_info_static(), v1::VariadicSplit::get_type_info_static()},
        DisabledFusion{pass::RoPEFusionGPTNEOX::get_type_info_static(), v1::VariadicSplit::get_type_info_static()},
        DisabledFusion{pass::RMSFusion::get_type_info_static(), v1::ReduceMean::get_type_info_static()},
        DisabledFusion{pass::RMSFusionMatcher::get_type_info_static(), v1::ReduceMean::get_type_info_static()},
        DisabledFusion{pass::FuseQSAIndexerQueryRoPE<op_internal::PagedQSAIndexer>::get_type_info_static(),
                       op_internal::RoPE::get_type_info_static()},
        DisabledFusion{pass::FuseQSAIndexerQueryNorm<op_internal::PagedQSAIndexer>::get_type_info_static(),
                       op_internal::RMS::get_type_info_static()},
        DisabledFusion{pass::FuseQSAIndexerQuery::get_type_info_static(), op_internal::RMS::get_type_info_static()},
        DisabledFusion{pass::FuseQSAIndexer::get_type_info_static(),
                       v17::SparseAttentionIndexer::get_type_info_static(),
                       true},
        DisabledFusion{pass::PagedQSAIndexerFusion::get_type_info_static(),
                       op_internal::QSAIndexer::get_type_info_static(),
                       true}),
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
    EXPECT_NO_THROW(manager.run_passes(model));

    EXPECT_EQ(callback_count, 0);
    run_query_compilation_passes(model, manager.get_pass_config());
    EXPECT_GT(callback_count, 0);
    EXPECT_EQ(count_ops(model, v1::ReduceMean::get_type_info_static()), 1);
    EXPECT_EQ(count_ops(model, op_internal::RMS::get_type_info_static()), 0);
    EXPECT_EQ(count_ops(model, op_internal::PagedQSAIndexer::get_type_info_static()), 1);
    EXPECT_EQ(count_ops(model, op_internal::PagedSparseAttention::get_type_info_static()), 1);
}

TEST(PagedQSAIndexerFusion, SDPAToPagedAttentionKeepsQueryWithoutRoPE) {
    const auto model = make_sparse_attention_model(false);
    pass::Manager manager;
    manager.register_pass<pass::SDPAToPagedAttention>();
    EXPECT_NO_THROW(manager.run_passes(model));
    EXPECT_EQ(count_ops(model, op_internal::PagedQSAIndexer::get_type_info_static()), 1);
    EXPECT_EQ(count_ops(model, op_internal::RMS::get_type_info_static()), 1);
    for (const auto& node : model->get_ops()) {
        if (const auto indexer = ov::as_type_ptr<op_internal::PagedQSAIndexer>(node)) {
            EXPECT_FALSE(indexer->get_config().query_norm);
            EXPECT_FALSE(indexer->get_config().query_rope);
        }
    }
}

}  // namespace
