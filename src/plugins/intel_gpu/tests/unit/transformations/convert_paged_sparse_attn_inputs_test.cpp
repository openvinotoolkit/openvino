// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/convert_paged_sparse_attn_inputs.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"

#include "intel_gpu/op/sparse_kv_quant.hpp"
#include "intel_gpu/op/sparse_pa_config.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/paged_sparse_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/manager.hpp"

namespace {

using Pass = ov::intel_gpu::ConvertPagedSparseAttnInputs;
using In = ov::intel_gpu::op::SparsePAInputIdx;
namespace v0 = ov::op::v0;

constexpr int64_t heads = 8;
constexpr int64_t kv_heads = 2;
constexpr int64_t key_block_size = 4;
constexpr int64_t topk = 16;
constexpr size_t page_size = 16;

struct Geometry {
    int64_t k_head_size = 64;
    int64_t v_head_size = 64;
    bool rt_info = true;  // num_k_heads / k_head_size / num_v_heads / v_head_size on the op
};

// PagedSparseAttention as the paged-attention conversion emits it: caches with open element type and shape.
struct Graph {
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<v0::Parameter> key_cache, value_cache;
};

Graph make_graph(const Geometry& g, const ov::PartialShape& cache_shape = ov::PartialShape::dynamic(4), ov::element::Type cache_type = ov::element::dynamic) {
    ov::ParameterVector params;
    ov::OutputVector args;
    const auto param = [&](ov::element::Type type, const ov::PartialShape& shape) {
        params.push_back(std::make_shared<v0::Parameter>(type, shape));
        args.push_back(params.back());
        return params.back();
    };
    param(ov::element::f16, {-1, heads * g.k_head_size});
    param(ov::element::f16, {-1, kv_heads * g.k_head_size});
    param(ov::element::f16, {-1, kv_heads * g.v_head_size});
    Graph graph;
    graph.key_cache = param(cache_type, cache_shape);
    graph.value_cache = param(cache_type, cache_shape);
    for (int i = 0; i < 4; ++i) {  // past_lens, subsequence_begins, block_indices, block_indices_begins
        param(ov::element::i32, {-1});
    }
    param(ov::element::i32, {-1, 1, topk});
    param(ov::element::i32, {-1, 1});
    args.push_back(v0::Constant::create(ov::element::f16, ov::Shape{}, {0.125f}));
    args.push_back(v0::Constant::create(ov::element::f16, ov::Shape{0}, std::vector<float>{}));

    const auto attention = std::make_shared<ov::op::internal::PagedSparseAttention>(args, key_block_size);
    if (g.rt_info) {
        auto& rt = attention->get_rt_info();
        rt["num_k_heads"] = kv_heads;
        rt["k_head_size"] = g.k_head_size;
        rt["num_v_heads"] = kv_heads;
        rt["v_head_size"] = g.v_head_size;
    }
    graph.model = std::make_shared<ov::Model>(ov::OutputVector{attention}, params);
    return graph;
}

// The GPU pipeline registration: the PagedAttention instance of ConvertPagedAttnInputs skips PagedSparseAttention.
void register_passes(ov::pass::Manager& manager, const Graph& graph, ov::element::Type precision, bool key_by_channel = true) {
    const auto pass_config = manager.get_pass_config();
    pass_config->set_callback<ov::pass::ConvertPagedAttnInputs>([](const std::shared_ptr<const ov::Node>& node) {
        return ov::is_type<ov::op::internal::PagedSparseAttention>(node);
    });
    pass_config->set_callback<Pass>(Pass::skip);
    ov::pass::ConvertPagedAttnInputs::KVCacheConfig pa_config;
    pa_config.keyCachePrecision = pa_config.valueCachePrecision = precision;
    pa_config.inferencePrecision = ov::element::f16;
    pa_config.keyCacheBlockSize = pa_config.valueCacheBlockSize = page_size;
    pa_config.keyCacheDimOrder = {0, 1, 3, 2};
    manager.register_pass<ov::pass::ConvertPagedAttnInputs>(pa_config, [](const ov::element::Type, const bool, const size_t, int64_t&, int64_t&) {});

    Pass::CacheConfig cache_config;
    cache_config.precision = precision;
    cache_config.key_by_channel = key_by_channel;
    cache_config.page_size = page_size;
    cache_config.precision = Pass::select_precision(*graph.model, cache_config);
    manager.register_pass<Pass>(cache_config);
}

void run(const Graph& graph, ov::element::Type precision, bool key_by_channel = true) {
    ov::pass::Manager manager;
    register_passes(manager, graph, precision, key_by_channel);
    manager.run_passes(graph.model);
}

ov::PartialShape page_shape(const ov::intel_gpu::op::KVQuant& quant, int64_t head_size) {
    if (!quant.enabled()) {
        return {-1, kv_heads, static_cast<int64_t>(page_size), head_size};
    }
    const auto layout = ov::intel_gpu::op::kv_page_layout(quant, page_size, static_cast<size_t>(head_size));
    return {-1, kv_heads, static_cast<int64_t>(layout.total_rows), static_cast<int64_t>(layout.row_bytes)};
}

}  // namespace

// An f16 KV cache precision gives f16 [blocks, Hkv, page_size, head_size] caches (not the PagedAttention key order).
TEST(ConvertPagedSparseAttnInputsTest, LaysOutF16Caches) {
    Geometry g;
    g.v_head_size = 32;
    const auto graph = make_graph(g);
    run(graph, ov::element::f16);
    EXPECT_EQ(graph.key_cache->get_element_type(), ov::element::f16);
    EXPECT_EQ(graph.value_cache->get_element_type(), ov::element::f16);
    EXPECT_EQ(graph.key_cache->get_partial_shape(), (ov::PartialShape{-1, kv_heads, 16, 64}));
    EXPECT_EQ(graph.value_cache->get_partial_shape(), (ov::PartialShape{-1, kv_heads, 16, 32}));
}

// An 8-bit cache precision gives u8 SparsePA pages: the key per channel (groups of 16 tokens), the value per token.
// 64 channels x 8 bits: 64 bytes per row; one group of key parameters (2 x 64 f16) takes 4 rows, the 16 value
// (scale, zp) pairs 1 row.
TEST_F(TransformationTestsF, ConvertPagedSparseAttnInputsQuantizesCachesByChannel) {
    const Geometry g;
    const auto graph = make_graph(g);
    model = graph.model;
    register_passes(manager, graph, ov::element::i8);

    const auto ref = make_graph(g);
    ref.key_cache->set_element_type(ov::element::u8);
    ref.key_cache->set_partial_shape({-1, kv_heads, 20, 64});
    ref.value_cache->set_element_type(ov::element::u8);
    ref.value_cache->set_partial_shape({-1, kv_heads, 17, 64});
    ref.model->validate_nodes_and_infer_types();
    model_ref = ref.model;
}

TEST(ConvertPagedSparseAttnInputsTest, QuantizesCachesByToken4Bit) {
    const Geometry g;
    const auto graph = make_graph(g);
    run(graph, ov::element::u4, false);
    const auto quant = ov::intel_gpu::op::paged_cache_quant(ov::element::u4, false);
    EXPECT_EQ(quant.bits, 4);
    EXPECT_EQ(graph.key_cache->get_element_type(), ov::element::u8);
    EXPECT_EQ(graph.key_cache->get_partial_shape(), page_shape(quant, 64));
    EXPECT_EQ(graph.value_cache->get_partial_shape(), page_shape(quant, 64));
}

// SparsePA quantizes power-of-two head sizes of at least 32 only: the caches stay f16.
TEST(ConvertPagedSparseAttnInputsTest, KeepsF16CachesWhenQuantizationIsUnsupported) {
    Geometry g;
    g.k_head_size = g.v_head_size = 48;
    const auto graph = make_graph(g);
    run(graph, ov::element::i8);
    EXPECT_EQ(graph.key_cache->get_element_type(), ov::element::f16);
    EXPECT_EQ(graph.key_cache->get_partial_shape(), (ov::PartialShape{-1, kv_heads, 16, 48}));
}

// Without the head geometry the caches are taken as they are.
TEST(ConvertPagedSparseAttnInputsTest, KeepsCachesWithoutHeadGeometry) {
    Geometry g;
    g.rt_info = false;
    const ov::PartialShape cache_shape{-1, kv_heads, 32, 64};
    const auto graph = make_graph(g, cache_shape, ov::element::f16);
    run(graph, ov::element::i8);
    EXPECT_EQ(graph.key_cache->get_element_type(), ov::element::f16);
    EXPECT_EQ(graph.key_cache->get_partial_shape(), cache_shape);
}

TEST(ConvertPagedSparseAttnInputsTest, SkipsOtherOps) {
    const auto parameter = std::make_shared<v0::Parameter>(ov::element::f16, ov::PartialShape{1});
    EXPECT_TRUE(Pass::skip(parameter));
    const auto graph = make_graph(Geometry{});
    EXPECT_FALSE(Pass::skip(graph.model->get_results()[0]->get_input_node_shared_ptr(0)));
}
