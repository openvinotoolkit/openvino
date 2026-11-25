// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "intel_gpu/op/kv_cache.hpp"
#include "intel_gpu/op/qsa_indexer_in_place.hpp"
#include "intel_gpu/op/read_value.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/qsa_indexer.hpp"
#include "openvino/op/qsa_prepare.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sparse_attention_indexer.hpp"
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

}  // namespace

TEST(QSAIndexerBlockCacheFusion, beam_gathered_cache_becomes_indirect_kv_cache) {
    expect_in_place_block_cache(make_indexer_model({}), true);
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
