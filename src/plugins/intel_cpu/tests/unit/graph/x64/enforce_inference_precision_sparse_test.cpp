// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "graph.h"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/paged_qsa_indexer.hpp"
#include "openvino/op/paged_sparse_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

using namespace ov::intel_cpu;

namespace {

// A Convert to bf16 before every Result makes the outputs already match the inference precision, so the node under test
// is not part of the graph tail and EnforceInferencePrecision treats it like a node in the middle of a model.
std::shared_ptr<const ov::Model> make_model_with_bf16_results(const std::shared_ptr<ov::Node>& node,
                                                              const ov::ParameterVector& params) {
    ov::ResultVector results;
    for (const auto& output : node->outputs()) {
        results.push_back(
            std::make_shared<ov::op::v0::Result>(std::make_shared<ov::op::v0::Convert>(output, ov::element::bf16)));
    }
    return std::make_shared<const ov::Model>(results, params);
}

void expect_inputs_keep_precision(const Graph& graph, Type type, const std::vector<ov::element::Type>& expected) {
    NodePtr found;
    for (const auto& node : graph.GetNodes()) {
        if (node->getType() == type) {
            found = node;
        }
    }
    ASSERT_NE(found, nullptr);
    ASSERT_EQ(found->getOriginalInputsNumber(), expected.size());
    for (size_t i = 0; i < expected.size(); i++) {
        EXPECT_EQ(found->getOriginalInputPrecisionAtPort(i), expected[i]) << "input " << i;
    }
}

Graph init_graph_with_bf16_inference(const std::shared_ptr<const ov::Model>& model) {
    Config config;
    config.inferencePrecision = ov::element::bf16;
    config.inferencePrecisionSetExplicitly = true;
    auto context = std::make_shared<GraphContext>(config, nullptr, false);
    Graph graph;
    graph.Init(model, context);
    return graph;
}

// PagedSparseAttention with u8 caches (the default KV cache precision): f32 query, key, value and scale.
std::shared_ptr<ov::op::internal::PagedSparseAttention> make_sparse_pa(ov::ParameterVector& params) {
    const auto f32 = ov::element::f32;
    const auto i32 = ov::element::i32;
    const size_t tokens = 4;
    const size_t blocks = 2;
    auto p = [](ov::element::Type type, const ov::Shape& shape) {
        return std::make_shared<ov::op::v0::Parameter>(type, shape);
    };
    // u8 caches keep a scale and a zero point (2 f32 = 8 bytes): the key cache is quantized by channel (8 extra rows),
    // the value cache by token (8 extra elements per row)
    const ov::Shape key_cache{blocks, 2, 4 + 8, 16};
    const ov::Shape value_cache{blocks, 2, 4, 16 + 8};
    params = {p(f32, {tokens, 64}),
              p(f32, {tokens, 32}),
              p(f32, {tokens, 32}),
              p(ov::element::u8, key_cache),
              p(ov::element::u8, value_cache),
              p(i32, {1}),
              p(i32, {2}),
              p(i32, {blocks}),
              p(i32, {2}),
              p(i32, {tokens, 1, 4}),
              p(i32, {tokens, 1})};
    ov::OutputVector args(params.begin(), params.end());
    args.push_back(ov::op::v0::Constant::create(f32, ov::Shape{}, {0.25F}));
    args.push_back(ov::op::v0::Constant::create(f32, ov::Shape{0}, std::vector<float>{}));
    auto sparse_pa = std::make_shared<ov::op::internal::PagedSparseAttention>(args, 2);
    // The head counts and sizes the node takes from the op, as the transformation that creates it records them.
    sparse_pa->get_rt_info()["num_k_heads"] = size_t{2};
    sparse_pa->get_rt_info()["k_head_size"] = size_t{16};
    sparse_pa->get_rt_info()["num_v_heads"] = size_t{2};
    sparse_pa->get_rt_info()["v_head_size"] = size_t{16};
    return sparse_pa;
}

}  // namespace

// The caches are written in place, so their ports are never converted. With a quantized cache (the default KV cache
// precision) query, key, value and the constant scale follow the inference precision, as the op needs them to share
// one type.
TEST(EnforceInferencePrecisionBF16Test, SparsePAFollowsInferencePrecisionWithQuantizedCache) {
    const auto f32 = ov::element::f32;
    const auto bf16 = ov::element::bf16;
    const auto u8 = ov::element::u8;
    const auto i32 = ov::element::i32;
    ov::ParameterVector params;
    auto sparse_pa = make_sparse_pa(params);

    auto graph = init_graph_with_bf16_inference(make_model_with_bf16_results(sparse_pa, params));
    expect_inputs_keep_precision(graph,
                                 Type::SparsePA,
                                 {bf16, bf16, bf16, u8, u8, i32, i32, i32, i32, i32, i32, bf16, f32});
}

// The op needs one element type on all float inputs, so with f32 caches the whole node stays f32.
TEST(EnforceInferencePrecisionBF16Test, KeepsPagedQSAIndexerInputsInOriginalPrecision) {
    const auto f32 = ov::element::f32;
    const auto i32 = ov::element::i32;
    const size_t tokens = 4;
    const size_t blocks = 2;
    const size_t page = 4;
    const size_t head_dim = 16;
    const size_t rotary_dim = 8;
    auto p = [](ov::element::Type type, const ov::Shape& shape) {
        return std::make_shared<ov::op::v0::Parameter>(type, shape);
    };
    const ov::ParameterVector params{p(f32, {tokens, 2 * head_dim}),
                                     p(f32, {tokens, 2 * head_dim}),
                                     p(f32, {tokens, rotary_dim}),
                                     p(f32, {tokens, rotary_dim}),
                                     p(f32, {head_dim}),
                                     p(f32, {head_dim}),
                                     p(i32, {1}),
                                     p(i32, {2}),
                                     p(i32, {blocks}),
                                     p(i32, {2}),
                                     p(f32, {blocks, 2, page, head_dim}),
                                     p(f32, {blocks, page / 2, 2 * rotary_dim}),
                                     p(f32, {blocks, 2, page / 2, head_dim})};
    ov::op::internal::PagedQSAIndexer::Config config;
    config.key_block_size = 2;
    config.block_topk = 2;
    config.num_selection_heads = 1;
    config.indexer_rotary_dim = rotary_dim;
    config.indexer_head_dim = head_dim;
    config.epsilon = 1e-6F;
    config.scale = 0.25F;
    ov::OutputVector args(params.begin(), params.end());
    auto qsa = std::make_shared<ov::op::internal::PagedQSAIndexer>(args, config);

    auto graph = init_graph_with_bf16_inference(make_model_with_bf16_results(qsa, params));
    expect_inputs_keep_precision(graph,
                                 Type::PagedQSAIndexer,
                                 {f32, f32, f32, f32, f32, f32, i32, i32, i32, i32, f32, f32, f32});
}
