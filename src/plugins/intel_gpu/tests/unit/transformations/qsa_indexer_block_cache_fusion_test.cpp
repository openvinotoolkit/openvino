// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/kv_cache.hpp"
#include "intel_gpu/op/qsa_indexer_in_place.hpp"
#include "intel_gpu/op/read_value.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/qsa_indexer.hpp"
#include "openvino/op/qsa_prepare.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/sparse_attention_indexer.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/pass/manager.hpp"
#include "plugin/transformations/kv_cache_fusion.hpp"
#include "transformations/common_optimizations/fuse_qsa_indexer.hpp"

using namespace ov::intel_gpu;

namespace {

constexpr size_t head_dim = 64;
constexpr size_t rotary_dim = 32;
constexpr size_t key_block_size = 4;
constexpr size_t query_heads = 2;

struct Options {
    bool beam_gather = true;
    bool block_cache_is_output = false;
};

/// One exported QSA indexer (f16, as after ConvertPrecision) with its three states carried by ReadValue / Assign, lowered
/// by FuseQSAIndexer(padding = true) as the GPU pipeline does.
std::shared_ptr<ov::Model> make_indexer_model(const Options& options) {
    const auto f16 = ov::element::f16;
    const auto param = [](ov::element::Type type, const ov::PartialShape& shape) {
        return std::make_shared<ov::op::v0::Parameter>(type, shape);
    };
    const auto index_query = param(f16, {-1, query_heads, -1, head_dim});
    const auto raw_keys = param(f16, {-1, -1, head_dim});
    const auto cos = param(f16, {-1, -1, rotary_dim});
    const auto sin = param(f16, {-1, -1, rotary_dim});
    const auto attention_mask = param(ov::element::i64, {-1, -1});
    const auto beam_idx = param(ov::element::i32, {-1});
    ov::ParameterVector params{index_query, raw_keys, cos, sin, attention_mask, beam_idx};

    ov::SinkVector sinks;
    std::vector<std::shared_ptr<ov::op::util::Variable>> variables;
    const auto past_state = [&](const ov::PartialShape& shape, const char* name) -> ov::Output<ov::Node> {
        variables.push_back(std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{shape, f16, name}));
        const auto past = std::make_shared<ov::op::v6::ReadValue>(variables.back());
        if (!options.beam_gather) {
            return past;
        }
        return std::make_shared<ov::op::v8::Gather>(past, beam_idx, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    };
    const auto block_keys = past_state({-1, -1, head_dim}, "block_keys");
    const auto incomplete_keys = past_state({-1, key_block_size - 1, head_dim}, "incomplete_keys");
    const auto incomplete_rope = past_state({-1, 2 * rotary_dim}, "incomplete_rope");

    const auto key_length = std::make_shared<ov::op::v1::ReduceSum>(attention_mask, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}), false);
    const auto prepare = std::make_shared<ov::op::v17::QSAPrepare>(raw_keys,
                                                                   block_keys,
                                                                   key_length,
                                                                   cos,
                                                                   sin,
                                                                   ov::op::v0::Constant::create(f16, ov::Shape{head_dim}, std::vector<float>(head_dim, 1.0f)),
                                                                   incomplete_keys,
                                                                   incomplete_rope,
                                                                   static_cast<int64_t>(key_block_size));
    const auto indexer = std::make_shared<ov::op::v17::SparseAttentionIndexer>(index_query,
                                                                               prepare->output(0),
                                                                               key_length,
                                                                               static_cast<int64_t>(key_block_size),
                                                                               8 * key_block_size,
                                                                               1);
    for (size_t i = 0; i < 3; i++) {
        sinks.push_back(std::make_shared<ov::op::v6::Assign>(prepare->output(i), variables[i]));
    }
    ov::OutputVector results{indexer->output(0), indexer->output(1)};
    if (options.block_cache_is_output) {
        results.push_back(prepare->output(0));
    }
    const auto model = std::make_shared<ov::Model>(results, sinks, params);

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::FuseQSAIndexer>(true);
    manager.run_passes(model);
    return model;
}

template <typename T>
std::vector<std::shared_ptr<T>> ops_of_type(const std::shared_ptr<ov::Model>& model) {
    std::vector<std::shared_ptr<T>> ops;
    for (const auto& op : model->get_ordered_ops()) {
        if (const auto typed = ov::as_type_ptr<T>(op)) {
            ops.push_back(typed);
        }
    }
    return ops;
}

void run_kv_cache_fusion(const std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager;
    manager.register_pass<KVCacheFusion>();
    manager.run_passes(model);
    model->validate_nodes_and_infer_types();
}

void expect_in_place_block_cache(const std::shared_ptr<ov::Model>& model, bool indirect) {
    ASSERT_EQ(ops_of_type<ov::op::internal::QSAIndexer>(model).size(), 1u);
    run_kv_cache_fusion(model);

    EXPECT_TRUE(ops_of_type<ov::op::internal::QSAIndexer>(model).empty());
    const auto in_place = ops_of_type<op::QSAIndexerInPlace>(model);
    ASSERT_EQ(in_place.size(), 1u);
    const auto kv_cache = ov::as_type_ptr<op::KVCache>(in_place[0]->get_input_node_shared_ptr(op::QSAIndexerInPlace::InputIdx::BLOCK_INDEX_KEYS_CACHE));
    ASSERT_NE(kv_cache, nullptr);
    EXPECT_EQ(kv_cache->get_variable_id(), "block_keys");
    EXPECT_EQ(kv_cache->get_concat_axis(), 1);
    EXPECT_EQ(kv_cache->get_indirect(), indirect);
    EXPECT_EQ(kv_cache->get_rt_info().count(op::KV_CACHE_REQUIRES_IDENTITY_BEAM_IDX), indirect ? 1u : 0u);
    EXPECT_NE(ov::as_type_ptr<op::ReadValue>(kv_cache->get_input_node_shared_ptr(0)), nullptr);
    EXPECT_EQ(in_place[0]->get_output_partial_shape(op::QSAIndexerInPlace::OutputIdx::BLOCK_INDEX_KEYS_CACHE), ov::PartialShape{0});

    // Only the incomplete-block states keep their Assign.
    std::vector<std::string> assigned;
    for (const auto& sink : model->get_sinks()) {
        assigned.push_back(ov::as_type_ptr<ov::op::v6::Assign>(sink)->get_variable_id());
    }
    EXPECT_EQ(assigned.size(), 2u);
    EXPECT_EQ(std::count(assigned.begin(), assigned.end(), "block_keys"), 0);
}

/// The QSAIndexer inputs other than its block cache, as FuseQSAIndexer leaves them (one index key head).
struct IndexerInputs {
    ov::ParameterVector params;
    ov::OutputVector args;  ///< in QSAIndexer::InputIdx order; BLOCK_INDEX_KEYS_CACHE is left unset
};

IndexerInputs make_indexer_inputs() {
    using In = ov::op::internal::QSAIndexer::InputIdx;
    const auto f16 = ov::element::f16;
    IndexerInputs in;
    in.args.resize(In::COUNT);
    const auto param = [&in](size_t idx, ov::element::Type type, const ov::PartialShape& shape) {
        in.params.push_back(std::make_shared<ov::op::v0::Parameter>(type, shape));
        in.args[idx] = in.params.back();
    };
    param(In::INDEX_QUERY, f16, {-1, -1, query_heads, head_dim});
    param(In::RAW_INDEX_KEYS, f16, {-1, -1, 1, head_dim});
    param(In::ROPE_COS, f16, {-1, -1, rotary_dim});
    param(In::ROPE_SIN, f16, {-1, -1, rotary_dim});
    param(In::KEY_LENGTH, ov::element::i32, {-1});
    param(In::Q_NORM_WEIGHT, f16, {head_dim});
    param(In::K_NORM_WEIGHT, f16, {head_dim});
    param(In::PAST_INCOMPLETE_RAW_K, f16, {-1, 1, key_block_size - 1, head_dim});
    param(In::PAST_INCOMPLETE_ROPE, f16, {-1, 2 * rotary_dim});
    return in;
}

ov::op::internal::QSAIndexer::Config indexer_config() {
    ov::op::internal::QSAIndexer::Config config;
    config.key_block_size = key_block_size;
    config.block_topk = 8;
    config.indexer_rotary_dim = rotary_dim;
    config.indexer_head_dim = head_dim;
    return config;
}

ov::Output<ov::Node> i64_constant(const std::vector<int64_t>& values) {
    return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{values.size()}, values);
}

}  // namespace

// The beam-gathered [B, blocks, D_idx] cache, grown by the zero blocks and sliced back after the indexer, becomes an
// indirect KVCache the indexer updates in place; its beam_idx must be the identity.
TEST_F(TransformationTestsF, QSAIndexerBlockCacheFusionBeamGatheredCache) {
    using QSAIndexer = ov::op::internal::QSAIndexer;
    using In = QSAIndexer::InputIdx;
    using Out = QSAIndexer::OutputIdx;
    const auto f16 = ov::element::f16;
    const ov::op::util::VariableInfo cache_info{{-1, -1, head_dim}, f16, "block_keys"};
    {
        auto in = make_indexer_inputs();
        const auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
        const auto zero_blocks = std::make_shared<ov::op::v0::Parameter>(f16, ov::PartialShape{-1, 1, -1, head_dim});
        const auto block_count = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1});
        in.params.insert(in.params.end(), {beam_idx, zero_blocks, block_count});
        const auto variable = std::make_shared<ov::op::util::Variable>(cache_info);
        const auto past = std::make_shared<ov::op::v8::Gather>(std::make_shared<ov::op::v6::ReadValue>(variable), beam_idx, i64_constant({0}));
        const auto with_head = std::make_shared<ov::op::v0::Unsqueeze>(past, i64_constant({1}));
        in.args[In::BLOCK_INDEX_KEYS_CACHE] = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{with_head, zero_blocks}, 2);
        const auto indexer = std::make_shared<QSAIndexer>(in.args, indexer_config());
        const auto cache = std::make_shared<ov::op::v0::Squeeze>(indexer->output(Out::BLOCK_INDEX_KEYS_CACHE), i64_constant({1}));
        const auto sliced = std::make_shared<ov::op::v8::Slice>(cache, i64_constant({0}), block_count, i64_constant({1}), i64_constant({1}));
        model = std::make_shared<ov::Model>(ov::OutputVector{indexer->output(Out::SELECTED_BLOCK_INDICES), indexer->output(Out::SELECTED_BLOCK_COUNT)},
                                            ov::SinkVector{std::make_shared<ov::op::v6::Assign>(sliced, variable)},
                                            in.params);
        manager.register_pass<KVCacheFusion>();
    }
    {
        auto in = make_indexer_inputs();
        const auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
        const auto zero_blocks = std::make_shared<ov::op::v0::Parameter>(f16, ov::PartialShape{-1, 1, -1, head_dim});
        const auto block_count = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1});
        in.params.insert(in.params.end(), {beam_idx, zero_blocks, block_count});
        const auto variable = std::make_shared<ov::op::util::Variable>(cache_info);
        const auto new_blocks = std::make_shared<ov::op::v0::Squeeze>(zero_blocks, i64_constant({1}));
        const auto kv_cache = std::make_shared<op::KVCache>(std::make_shared<op::ReadValue>(variable), new_blocks, beam_idx, variable, 1, 0, f16);
        kv_cache->get_rt_info()[op::KV_CACHE_REQUIRES_IDENTITY_BEAM_IDX] = true;
        in.args[In::BLOCK_INDEX_KEYS_CACHE] = kv_cache->output(0);
        const auto in_place = std::make_shared<op::QSAIndexerInPlace>(in.args, indexer_config());
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{in_place->output(Out::SELECTED_BLOCK_INDICES), in_place->output(Out::SELECTED_BLOCK_COUNT)},
                                                in.params);
    }
    comparator.enable(FunctionsComparator::ATTRIBUTES);
}

TEST(QSAIndexerBlockCacheFusion, cache_without_gather_becomes_kv_cache) {
    Options options;
    options.beam_gather = false;
    expect_in_place_block_cache(make_indexer_model(options), false);
}

TEST(QSAIndexerBlockCacheFusion, cache_read_by_other_ops_is_kept) {
    Options options;
    options.block_cache_is_output = true;
    const auto model = make_indexer_model(options);
    run_kv_cache_fusion(model);
    EXPECT_EQ(ops_of_type<ov::op::internal::QSAIndexer>(model).size(), 1u);
    EXPECT_TRUE(ops_of_type<op::QSAIndexerInPlace>(model).empty());
    EXPECT_TRUE(ops_of_type<op::KVCache>(model).empty());
    EXPECT_EQ(model->get_sinks().size(), 3u);
}
