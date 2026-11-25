// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/convert_paged_qsa_indexer_inputs.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/paged_qsa_indexer.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"

namespace {

using Indexer = ov::op::internal::PagedQSAIndexer;
namespace v0 = ov::op::v0;

constexpr int64_t index_heads = 4;
constexpr int64_t head_dim = 128;
constexpr int64_t rotary = 64;
constexpr size_t key_block_size = 4;

struct Graph {
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<v0::Parameter> key_cache, rope_cache, summary_cache;
};

// The PagedQSAIndexer the paged-attention conversion emits: caches with the model precision and open shapes.
Graph make_graph(const ov::PartialShape& key_cache_shape = ov::PartialShape{-1, 1, -1, head_dim},
                 bool converted = false,
                 const ov::PartialShape& rope_cache_shape = ov::PartialShape{-1, -1, 2 * rotary},
                 const ov::PartialShape& summary_cache_shape = ov::PartialShape{-1, 1, -1, head_dim}) {
    ov::ParameterVector params;
    const auto param = [&params](ov::element::Type type, const ov::PartialShape& shape) {
        params.push_back(std::make_shared<v0::Parameter>(type, shape));
        return params.back();
    };
    ov::OutputVector args{param(ov::element::f16, {-1, index_heads * head_dim}),
                          param(ov::element::f16, {-1, head_dim}),
                          param(ov::element::f16, {-1, rotary}),
                          param(ov::element::f16, {-1, rotary}),
                          param(ov::element::f16, {head_dim}),
                          param(ov::element::f16, {head_dim})};
    for (int i = 0; i < 4; ++i) {  // past_lens, subsequence_begins, block_indices, block_indices_begins
        args.push_back(param(ov::element::i32, {-1}));
    }
    Graph g;
    // The indexer takes one floating-point type: f32 caches (the model precision) come through a Convert.
    const auto cache_type = converted ? ov::element::f32 : ov::element::f16;
    g.key_cache = param(cache_type, key_cache_shape);
    g.rope_cache = param(cache_type, rope_cache_shape);
    g.summary_cache = param(cache_type, summary_cache_shape);
    for (const auto& cache : {g.key_cache, g.rope_cache, g.summary_cache}) {
        // As ConvertPrecision leaves a keep_const_precision cache for an f16 model.
        args.push_back(converted ? std::make_shared<v0::Convert>(cache, ov::element::f16)->output(0) : cache->output(0));
    }
    Indexer::Config config;
    config.key_block_size = key_block_size;
    config.block_topk = 512;
    config.indexer_rotary_dim = rotary;
    config.indexer_head_dim = head_dim;
    const auto indexer = std::make_shared<Indexer>(args, config);
    g.model = std::make_shared<ov::Model>(indexer->outputs(), params);
    return g;
}

void run(const Graph& g, size_t page_size = 16) {
    ov::pass::Manager manager;
    manager.register_pass<ov::intel_gpu::ConvertPagedQSAIndexerInputs>(page_size);
    manager.run_passes(g.model);
}

}  // namespace

// The caches become f16 pages of the paged-attention block size, and the Convert ConvertPrecision puts in front of
// them is removed: the indexer updates the caches.
TEST_F(TransformationTestsF, ConvertPagedQSAIndexerInputsLaysOutCaches) {
    model = make_graph({-1, 1, -1, head_dim}, true).model;
    manager.register_pass<ov::intel_gpu::ConvertPagedQSAIndexerInputs>(16);
    model_ref = make_graph({-1, 1, 16, head_dim}, false, {-1, 4, 2 * rotary}, {-1, 1, 4, head_dim}).model;
}

// A page size declared by the index key cache is kept.
TEST(ConvertPagedQSAIndexerInputsTest, KeepsDeclaredPageSize) {
    const auto g = make_graph({-1, 1, 32, head_dim});
    run(g);
    EXPECT_EQ(g.key_cache->get_partial_shape(), (ov::PartialShape{-1, 1, 32, head_dim}));
    EXPECT_EQ(g.summary_cache->get_partial_shape(), (ov::PartialShape{-1, 1, 8, head_dim}));
    EXPECT_EQ(g.summary_cache->get_element_type(), ov::element::f16);
}

// The caches keep their f16 precision through the later ConvertPrecision (not compared by the reference graph above).
TEST(ConvertPagedQSAIndexerInputsTest, KeepsCachePrecision) {
    const auto g = make_graph({-1, 1, -1, head_dim}, true);
    run(g);
    for (const auto& cache : {g.key_cache, g.rope_cache, g.summary_cache}) {
        EXPECT_TRUE(ov::is_keep_const_precision(cache));
    }
}
