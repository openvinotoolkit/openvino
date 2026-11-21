// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/convert_paged_sparse_attention_to_sparse_pa.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/sparse_pa.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/paged_sparse_attention.hpp"
#include "openvino/op/parameter.hpp"

using namespace testing;

namespace {

using SparsePA = ov::intel_gpu::op::SparsePA;
using Pass = ov::intel_gpu::ConvertPagedSparseAttentionToSparsePA;
namespace v0 = ov::op::v0;

constexpr int64_t heads = 8;
constexpr int64_t kv_heads = 2;
constexpr int64_t key_block_size = 4;
constexpr int64_t topk = 16;

struct Geometry {
    int64_t k_head_size = 64;
    int64_t v_head_size = 64;
    int64_t query_width = heads * 64;
    bool rt_info = true;  // num_k_heads / k_head_size / num_v_heads / v_head_size on the op
};

// The 13 inputs of PagedSparseAttention; the caches as given by `key_cache_shape` / `value_cache_shape`.
struct Inputs {
    ov::ParameterVector params;
    ov::OutputVector args;
};

Inputs make_inputs(const Geometry& g,
                   const ov::PartialShape& key_cache_shape = ov::PartialShape::dynamic(4),
                   const ov::PartialShape& value_cache_shape = ov::PartialShape::dynamic(4),
                   ov::element::Type cache_type = ov::element::f16) {
    Inputs in;
    const auto param = [&in](ov::element::Type type, const ov::PartialShape& shape) {
        in.params.push_back(std::make_shared<v0::Parameter>(type, shape));
        in.args.push_back(in.params.back());
    };
    param(ov::element::f16, {-1, g.query_width});
    param(ov::element::f16, {-1, kv_heads * g.k_head_size});
    param(ov::element::f16, {-1, kv_heads * g.v_head_size});
    param(cache_type, key_cache_shape);
    param(cache_type, value_cache_shape);
    for (int i = 0; i < 4; ++i) {  // past_lens, subsequence_begins, block_indices, block_indices_begins
        param(ov::element::i32, {-1});
    }
    param(ov::element::i32, {-1, 1, topk});
    param(ov::element::i32, {-1, 1});
    in.args.push_back(v0::Constant::create(ov::element::f16, ov::Shape{}, {0.125f}));
    in.args.push_back(v0::Constant::create(ov::element::f16, ov::Shape{0}, std::vector<float>{}));
    return in;
}

std::shared_ptr<ov::Model> make_model(const Inputs& in, const Geometry& g) {
    const auto attention = std::make_shared<ov::op::internal::PagedSparseAttention>(in.args, key_block_size);
    if (g.rt_info) {
        auto& rt = attention->get_rt_info();
        rt["num_k_heads"] = static_cast<size_t>(kv_heads);
        rt["k_head_size"] = static_cast<size_t>(g.k_head_size);
        rt["num_v_heads"] = static_cast<size_t>(kv_heads);
        rt["v_head_size"] = static_cast<size_t>(g.v_head_size);
    }
    return std::make_shared<ov::Model>(ov::OutputVector{attention}, in.params);
}

SparsePA::Config sparse_config(const Geometry& g, size_t page_size) {
    SparsePA::Config config;
    config.num_heads = static_cast<size_t>(g.query_width / g.k_head_size);
    config.num_kv_heads = kv_heads;
    config.num_selection_heads = 1;
    config.k_head_size = static_cast<size_t>(g.k_head_size);
    config.v_head_size = static_cast<size_t>(g.v_head_size);
    config.key_block_size = key_block_size;
    config.page_size = page_size;
    return config;
}

std::shared_ptr<ov::Model> make_ref(const Inputs& in, const SparsePA::Config& config) {
    const auto sparse_pa = std::make_shared<SparsePA>(in.args, config);
    return std::make_shared<ov::Model>(ov::OutputVector{sparse_pa}, in.params);
}

class ConvertPagedSparseAttentionToSparsePATest : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        comparator.enable(FunctionsComparator::ATTRIBUTES);
        comparator.enable(FunctionsComparator::CONST_VALUES);
        comparator.enable(FunctionsComparator::PRECISIONS);
    }
    void register_pass(ov::element::Type precision = ov::element::f16) {
        Pass::CacheConfig cache;
        cache.precision = precision;
        manager.register_pass<Pass>(cache);
    }
};

}  // namespace

// The paged-attention conversion leaves the cache shapes open: they become f16 pages of the configured size.
TEST_F(ConvertPagedSparseAttentionToSparsePATest, LaysOutF16CachesFromRuntimeInfo) {
    const Geometry g;
    model = make_model(make_inputs(g), g);
    register_pass();
    model_ref = make_ref(make_inputs(g, {-1, kv_heads, 16, g.k_head_size}, {-1, kv_heads, 16, g.v_head_size}), sparse_config(g, 16));
}

// Without runtime info the KV head count is key_cache dim 1; caches declared with a page size keep it.
TEST_F(ConvertPagedSparseAttentionToSparsePATest, KeepsDeclaredPageSize) {
    Geometry g;
    g.rt_info = false;
    g.v_head_size = 32;
    const ov::PartialShape key_cache{-1, kv_heads, 32, g.k_head_size};
    const ov::PartialShape value_cache{-1, kv_heads, 32, g.v_head_size};
    model = make_model(make_inputs(g, key_cache, value_cache), g);
    register_pass(ov::element::i8);
    model_ref = make_ref(make_inputs(g, key_cache, value_cache), sparse_config(g, 32));
}

// An 8-bit cache precision gives u8 SparsePA pages: the key per channel (groups of 16 tokens), the value per token.
TEST_F(ConvertPagedSparseAttentionToSparsePATest, QuantizesCaches) {
    const Geometry g;
    model = make_model(make_inputs(g), g);
    register_pass(ov::element::i8);

    auto config = sparse_config(g, 16);
    config.key_quant.mode = ov::intel_gpu::op::KVQuant::Mode::BY_CHANNEL;
    config.key_quant.bits = 8;
    config.key_quant.group_size = 16;
    config.value_quant.mode = ov::intel_gpu::op::KVQuant::Mode::BY_TOKEN;
    config.value_quant.bits = 8;
    const auto key_layout = ov::intel_gpu::op::kv_page_layout(config.key_quant, 16, g.k_head_size);
    const auto value_layout = ov::intel_gpu::op::kv_page_layout(config.value_quant, 16, g.v_head_size);
    // 64 channels x 8 bits: 64 bytes per row; one group of key parameters (2 x 64 f16) takes 4 rows, the 16 value
    // (scale, zp) pairs 1 row.
    ASSERT_EQ(key_layout.total_rows, 20u);
    ASSERT_EQ(value_layout.total_rows, 17u);
    model_ref = make_ref(make_inputs(g,
                                     {-1, kv_heads, static_cast<int64_t>(key_layout.total_rows), static_cast<int64_t>(key_layout.row_bytes)},
                                     {-1, kv_heads, static_cast<int64_t>(value_layout.total_rows), static_cast<int64_t>(value_layout.row_bytes)},
                                     ov::element::u8),
                         config);
}

// SparsePA quantizes power-of-two head sizes of at least 32 only: other caches stay f16.
TEST_F(ConvertPagedSparseAttentionToSparsePATest, KeepsF16CachesWhenQuantizationIsUnsupported) {
    Geometry g;
    g.k_head_size = g.v_head_size = 48;
    g.query_width = heads * 48;
    model = make_model(make_inputs(g), g);
    register_pass(ov::element::i8);
    model_ref = make_ref(make_inputs(g, {-1, kv_heads, 16, 48}, {-1, kv_heads, 16, 48}), sparse_config(g, 16));
}

// Caches computed in the graph are not laid out by the pass: f16 ones are taken as they are.
TEST_F(ConvertPagedSparseAttentionToSparsePATest, TakesComputedF16Caches) {
    const Geometry g;
    const auto build = [&g](bool converted) {
        auto in = make_inputs(g, {-1, kv_heads, 16, g.k_head_size}, {-1, kv_heads, 16, g.v_head_size}, ov::element::f32);
        in.args[SparsePA::InputIdx::KEY_CACHE] = std::make_shared<v0::Convert>(in.args[SparsePA::InputIdx::KEY_CACHE], ov::element::f16);
        in.args[SparsePA::InputIdx::VALUE_CACHE] = std::make_shared<v0::Convert>(in.args[SparsePA::InputIdx::VALUE_CACHE], ov::element::f16);
        return converted ? make_ref(in, sparse_config(g, 0)) : make_model(in, g);
    };
    model = build(false);
    register_pass(ov::element::i8);
    model_ref = build(true);
}

// SparsePA derives from PagedSparseAttention but is already lowered.
TEST_F(ConvertPagedSparseAttentionToSparsePATest, KeepsSparsePA) {
    const Geometry g;
    model = make_ref(make_inputs(g, {-1, kv_heads, 16, g.k_head_size}, {-1, kv_heads, 16, g.v_head_size}), sparse_config(g, 16));
    register_pass();
}

// SparsePA is specialized for the head geometry.
TEST_F(ConvertPagedSparseAttentionToSparsePATest, KeepsDynamicQueryWidth) {
    Geometry g;
    g.query_width = -1;
    model = make_model(make_inputs(g), g);
    register_pass();
}

TEST_F(ConvertPagedSparseAttentionToSparsePATest, KeepsOpWithoutHeadGeometry) {
    Geometry g;
    g.rt_info = false;
    model = make_model(make_inputs(g), g);
    register_pass();
}

TEST_F(ConvertPagedSparseAttentionToSparsePATest, KeepsUnsupportedHeadSize) {
    Geometry g;
    g.k_head_size = g.v_head_size = 24;
    g.query_width = heads * 24;
    model = make_model(make_inputs(g), g);
    register_pass();
}
