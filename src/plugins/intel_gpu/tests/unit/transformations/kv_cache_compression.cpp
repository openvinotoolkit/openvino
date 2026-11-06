// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"

#include <functional>

#include "openvino/core/model.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"

#include "intel_gpu/op/read_value.hpp"
#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/read_values.hpp"
#include "intel_gpu/op/kv_cache.hpp"
#include "intel_gpu/op/kv_cache_compressed.hpp"
#include "intel_gpu/op/indirect_sdpa.hpp"

#include "plugin/transformations/kv_cache_compression.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

// Pass and model configuration. The K-bias gate only applies to the CVS-190235 scope: f16 compute, int8 KV-cache,
// non-IMMAD device and statically known grouped-query attention.
struct KBiasModelConfig {
    // Query (compute) precision; the gate only applies to f16.
    ov::element::Type query_dt = ov::element::f16;
    // Query heads != KV heads.
    bool gqa = true;
    // KV-cache compression precision passed to the pass.
    ov::element::Type compression_dt = ov::element::i8;
    bool supports_immad = false;
    // Use the [batch, seq, heads, head_size] layout (order {0, 2, 1, 3}) for query, key and value.
    bool transposed_qkv = false;
    // Query head count is not statically known.
    bool dynamic_heads = false;
};

struct KBiasTestCase {
    std::string name;
    bool compressed;
    // Model scenario to build; defaults to `name`.
    std::string scenario = {};
    KBiasModelConfig config = {};
};

namespace {
KBiasModelConfig make_config(const std::function<void(KBiasModelConfig&)>& modify) {
    KBiasModelConfig config;
    modify(config);
    return config;
}

std::shared_ptr<ov::Model> make_k_bias_model(const std::string& test_case, bool compressed, const KBiasModelConfig& cfg) {
    const auto query_dt = cfg.query_dt;
    const ov::Shape token_shape{1, 1, 1, 4};
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, token_shape);
    auto weights = ov::op::v0::Constant::create(ov::element::f16, {4, 4}, {1.f});
    auto projection = [&](float bias_value) -> std::shared_ptr<ov::Node> {
        auto linear = std::make_shared<ov::op::v0::MatMul>(data, weights, false, true);
        auto bias = ov::op::v0::Constant::create(ov::element::f16, {4}, {bias_value});
        return std::make_shared<ov::op::v1::Add>(linear, bias);
    };

    std::shared_ptr<ov::Node> key = projection(1.f);
    if (test_case == "LargeBias") {
        key = projection(200.f);
    } else if (test_case == "NegativeLargeBias") {
        key = projection(-200.f);
    } else if (test_case == "ZeroBias") {
        key = projection(0.f);
    } else if (test_case == "ThresholdBias") {
        key = projection(128.f);
    } else if (test_case == "FullyConnectedBias") {
        auto bias = ov::op::v0::Constant::create(ov::element::f16, {4}, {200.f});
        key = std::make_shared<ov::intel_gpu::op::FullyConnected>(data, weights, bias);
    } else if (test_case == "ConvertedBias") {
        auto bias = ov::op::v0::Constant::create(ov::element::f32, {4}, {200.f});
        auto converted = std::make_shared<ov::op::v0::Convert>(bias, ov::element::f16);
        auto linear = std::make_shared<ov::op::v0::MatMul>(data, weights, false, true);
        key = std::make_shared<ov::op::v1::Add>(linear, converted);
    } else if (test_case == "MultipleBranches" || test_case == "ReversedBranches") {
        ov::OutputVector branches{projection(200.f), key};
        if (test_case == "ReversedBranches") {
            std::swap(branches[0], branches[1]);
        }
        key = std::make_shared<ov::op::v0::Concat>(branches, 3);
    } else if (test_case == "SharedNodesDoNotConsumeBudget") {
        ov::OutputVector branches(80, key);
        branches[0] = projection(200.f);
        key = std::make_shared<ov::op::v0::Concat>(branches, 3);
    } else if (test_case == "RoPEAuxiliaryBias" || test_case == "RoPEDataBias") {
        auto cos = projection(200.f);
        auto sin = ov::op::v0::Constant::create(ov::element::f16, token_shape, {0.f});
        if (test_case == "RoPEDataBias") {
            key = projection(200.f);
        }
        ov::op::internal::RoPE::Config config;
        config.rotary_ndims = 4;
        key = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{key, cos, sin}, config);
    } else if (test_case == "GatherAuxiliaryBias") {
        auto indices = std::make_shared<ov::op::v0::Convert>(projection(200.f), ov::element::i32);
        auto axis = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
        auto gathered = std::make_shared<ov::op::v8::Gather>(key, indices, axis);
        auto shape = ov::op::v0::Constant::create(ov::element::i64, {4}, {1, 1, 4, 4});
        key = std::make_shared<ov::op::v1::Reshape>(gathered, shape, false);
    } else if (test_case == "UnsupportedPath") {
        key = std::make_shared<ov::op::v0::Abs>(projection(200.f));
    } else if (test_case == "BudgetExhausted") {
        key = projection(200.f);
        auto shape = ov::op::v0::Constant::create(ov::element::i64, {4}, {1, 1, 1, 4});
        for (size_t i = 0; i < 65; ++i) {
            key = std::make_shared<ov::op::v1::Reshape>(key, shape, false);
        }
    } else if (test_case == "NoProjection") {
        key = data;
    } else if (test_case == "SplitSelectedChannels") {
        // K is one slice of a fused projection whose bias is large in another (e.g. Q) slice.
        auto axis = ov::op::v0::Constant::create(ov::element::i64, {}, {3});
        key = std::make_shared<ov::op::v1::Split>(projection(200.f), axis, 2);
    } else if (test_case == "AddOfProjections") {
        // Not "linear + constant": the Add of two projections is unknown, not a supported multi-branch graph.
        key = std::make_shared<ov::op::v1::Add>(projection(200.f), projection(1.f));
    }

    auto shape = key->get_output_partial_shape(0);
    // Sequence axis: dim 2 for [batch, heads, seq, head_size], dim 1 for [batch, seq, heads, head_size].
    const size_t seq_dim = cfg.transposed_qkv ? 1 : 2;
    const size_t heads_dim = cfg.transposed_qkv ? 2 : 1;
    auto past_shape = shape;
    past_shape[seq_dim] = ov::Dimension::dynamic();
    // KV cache has a single head; the query has two heads for GQA.
    auto query_shape = shape;
    query_shape[heads_dim] = cfg.dynamic_heads ? ov::Dimension::dynamic() : ov::Dimension(cfg.gqa ? 2 : 1);
    auto query = std::make_shared<ov::op::v0::Parameter>(query_dt, query_shape);
    auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, shape);
    auto beam = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
    const std::vector<int64_t> qkv_order = cfg.transposed_qkv ? std::vector<int64_t>{0, 2, 1, 3} : std::vector<int64_t>{0, 1, 2, 3};
    const std::vector<int64_t> out_order{0, 1, 2, 3};
    auto key_variable = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{past_shape, compressed ? ov::element::i8 : ov::element::f16, "key"});
    auto value_variable = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{past_shape, compressed ? ov::element::i8 : ov::element::f16, "value"});

    std::shared_ptr<ov::Node> sdpa;
    if (compressed) {
        const bool planar = cfg.supports_immad;
        const bool is_int4 = cfg.compression_dt.bitwidth() == 4;
        ov::op::internal::DynamicQuantize::Attributes config;
        config.quantization_type = ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric;
        config.quantization_dt = ov::element::i8;
        config.scale_dt = query_dt;
        config.zp_dt = (cfg.supports_immad && !is_int4) ? ov::element::i8 : query_dt;
        config.group_sizes = {1, 1, 1, UINT64_MAX};
        config.scales_zp_output_order = std::vector<uint64_t>(qkv_order.begin(), qkv_order.end());
        config.output_storage_type = planar ? ov::op::internal::DynamicQuantize::OutputStorageType::Planar
                                            : ov::op::internal::DynamicQuantize::OutputStorageType::InterleavedScalesZP;
        ov::op::internal::DynamicQuantize dq;
        dq.set_attrs(config);
        const auto scale_shape = ov::op::internal::DynamicQuantize::shape_infer(&dq, {past_shape})[1];
        auto make_cache = [&](const std::shared_ptr<ov::op::util::Variable>& variable, const ov::Output<ov::Node>& token) {
            const auto& id = variable->get_info().variable_id;
            std::vector<ov::op::util::VariableInfo> infos{{past_shape, ov::element::i8, id}, {scale_shape, query_dt, id}};
            if (planar) {
                infos.push_back({scale_shape, config.zp_dt, id});
            }
            auto past = std::make_shared<ov::intel_gpu::op::ReadValues>(variable, infos);
            ov::OutputVector inputs{past->output(0), token, beam, past->output(1)};
            if (planar) {
                inputs.push_back(past->output(2));
            }
            return std::make_shared<ov::intel_gpu::op::KVCacheCompressed>(inputs, variable, false, seq_dim, 0, config);
        };
        auto key_cache = make_cache(key_variable, key);
        auto value_cache = make_cache(value_variable, value);
        ov::OutputVector sdpa_inputs{query, key_cache->output(0), value_cache->output(0), key_cache->output(2), value_cache->output(2)};
        if (planar) {
            sdpa_inputs.push_back(key_cache->output(3));
            sdpa_inputs.push_back(value_cache->output(3));
        }
        sdpa = std::make_shared<ov::intel_gpu::op::IndirectSDPA>(
            sdpa_inputs, key_cache->output(1), false, 0, qkv_order, qkv_order, qkv_order, out_order, config);
    } else {
        auto key_past = std::make_shared<ov::intel_gpu::op::ReadValue>(key_variable);
        auto value_past = std::make_shared<ov::intel_gpu::op::ReadValue>(value_variable);
        auto key_cache = std::make_shared<ov::intel_gpu::op::KVCache>(key_past, key, beam, key_variable, seq_dim, 0);
        auto value_cache = std::make_shared<ov::intel_gpu::op::KVCache>(value_past, value, beam, value_variable, seq_dim, 0);
        sdpa = std::make_shared<ov::intel_gpu::op::IndirectSDPA>(
            ov::OutputVector{query, key_cache->output(0), value_cache->output(0)},
            key_cache->output(1), false, 0, qkv_order, qkv_order, qkv_order, out_order);
    }
    return std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{data, query, value, beam});
}
}  // namespace

class KVCacheCompressionKBiasTest : public TransformationTestsF, public testing::WithParamInterface<KBiasTestCase> {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        comparator.enable(FunctionsComparator::ATTRIBUTES);
        comparator.enable(FunctionsComparator::CONST_VALUES);
    }
};

TEST_P(KVCacheCompressionKBiasTest, ProjectionDataPaths) {
    const auto& param = GetParam();
    const auto& scenario = param.scenario.empty() ? param.name : param.scenario;
    manager.register_pass<KVCacheCompression>(param.config.compression_dt, param.config.supports_immad);
    model = make_k_bias_model(scenario, false, param.config);
    // Build the reference explicitly: the fixture would otherwise clone `model`, and cloning an uncompressed
    // IndirectSDPA is not supported. An unchanged graph is expected when compression is skipped.
    model_ref = make_k_bias_model(scenario, param.compressed, param.config);
}

INSTANTIATE_TEST_SUITE_P(
    KVCacheCompression,
    KVCacheCompressionKBiasTest,
    testing::Values(KBiasTestCase{"SmallBias", true},
                    KBiasTestCase{"ZeroBias", true},
                    KBiasTestCase{"ThresholdBias", true},
                    KBiasTestCase{"LargeBias", false},
                    KBiasTestCase{"NegativeLargeBias", false},
                    KBiasTestCase{"FullyConnectedBias", false},
                    KBiasTestCase{"ConvertedBias", false},
                    KBiasTestCase{"MultipleBranches", false},
                    KBiasTestCase{"ReversedBranches", false},
                    KBiasTestCase{"SharedNodesDoNotConsumeBudget", false},
                    KBiasTestCase{"RoPEAuxiliaryBias", true},
                    KBiasTestCase{"RoPEDataBias", false},
                    KBiasTestCase{"GatherAuxiliaryBias", true},
                    KBiasTestCase{"UnsupportedPath", true},
                    KBiasTestCase{"BudgetExhausted", true},
                    KBiasTestCase{"NoProjection", true},
                    // Channel-selecting ops cannot be mapped to the bias statically: unknown, compression is kept.
                    KBiasTestCase{"SplitSelectedChannels", true},
                    // Only "linear + constant" Adds are understood; other Adds are unknown, compression is kept.
                    KBiasTestCase{"AddOfProjections", true},
                    // In scope (f16, int8/uint8, non-IMMAD, GQA): a large bias skips compression.
                    KBiasTestCase{"LargeBiasU8", false, "LargeBias",
                                  make_config([](KBiasModelConfig& c) { c.compression_dt = ov::element::u8; })},
                    KBiasTestCase{"LargeBiasTransposedGQA", false, "LargeBias",
                                  make_config([](KBiasModelConfig& c) { c.transposed_qkv = true; })},
                    // Outside the CVS-190235 scope: compression is kept despite the large bias.
                    KBiasTestCase{"LargeBiasF32Query", true, "LargeBias",
                                  make_config([](KBiasModelConfig& c) { c.query_dt = ov::element::f32; })},
                    KBiasTestCase{"LargeBiasNonGQA", true, "LargeBias",
                                  make_config([](KBiasModelConfig& c) { c.gqa = false; })},
                    KBiasTestCase{"LargeBiasIMMAD", true, "LargeBias",
                                  make_config([](KBiasModelConfig& c) { c.supports_immad = true; })},
                    KBiasTestCase{"LargeBiasInt4", true, "LargeBias",
                                  make_config([](KBiasModelConfig& c) { c.compression_dt = ov::element::i4; })},
                    KBiasTestCase{"LargeBiasUint4", true, "LargeBias",
                                  make_config([](KBiasModelConfig& c) { c.compression_dt = ov::element::u4; })},
                    KBiasTestCase{"LargeBiasIMMADInt4", true, "LargeBias",
                                  make_config([](KBiasModelConfig& c) {
                                      c.supports_immad = true;
                                      c.compression_dt = ov::element::i4;
                                  })},
                    KBiasTestCase{"LargeBiasTransposedNonGQA", true, "LargeBias",
                                  make_config([](KBiasModelConfig& c) {
                                      c.transposed_qkv = true;
                                      c.gqa = false;
                                  })},
                    KBiasTestCase{"LargeBiasDynamicHeads", true, "LargeBias",
                                  make_config([](KBiasModelConfig& c) { c.dynamic_heads = true; })}),
    [](const testing::TestParamInfo<KBiasTestCase>& info) { return info.param.name; });

TEST_F(TransformationTestsF, KVCacheCompression) {
    bool causal = false;
    bool with_mask = true;
    bool with_scale = true;
    size_t concat_axis = 2;
    size_t gather_axis = 0;
    ov::element::Type_t element_type = ov::element::f16;
    std::vector<int64_t> qkv_order = {0, 1, 2, 3};
    std::shared_ptr<ov::Node> mask = nullptr;
    std::shared_ptr<ov::Node> scale = nullptr;
    ov::PartialShape input_shape = ov::PartialShape{1, 32, -1, 80};

    {
        auto query = std::make_shared<ov::op::v0::Parameter>(element_type, input_shape);
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});

        auto key_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v0"});
        auto key_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto key_past = std::make_shared<ov::intel_gpu::op::ReadValue>(key_variable);
        auto key_cache = std::make_shared<ov::intel_gpu::op::KVCache>(key_past, key_current, beam_idx, key_variable, concat_axis, gather_axis);

        auto value_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v1"});
        auto value_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto value_past = std::make_shared<ov::intel_gpu::op::ReadValue>(value_variable);
        auto value_cache = std::make_shared<ov::intel_gpu::op::KVCache>(value_past, value_current, beam_idx, value_variable, concat_axis, gather_axis);

        ov::ParameterVector params{ beam_idx, query, key_current, value_current };

        if (with_mask) {
            auto attn_mask = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape::dynamic(4));
            mask = attn_mask;
            params.push_back(attn_mask);
        }

        if (with_mask && with_scale) {
            auto scale_input = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape{1});
            scale = scale_input;
            params.push_back(scale_input);
        }

        ov::OutputVector sdpa_inputs = { query, key_cache->output(0), value_cache->output(0) };

        if (mask) {
            sdpa_inputs.push_back(mask);
        }

        if (scale) {
            sdpa_inputs.push_back(scale);
        }

        std::shared_ptr<ov::Node> sdpa = nullptr;
        sdpa = std::make_shared<ov::intel_gpu::op::IndirectSDPA>(sdpa_inputs,
                                                                 key_cache->output(1),
                                                                 causal,
                                                                 gather_axis,
                                                                 qkv_order,
                                                                 qkv_order,
                                                                 qkv_order,
                                                                 ov::intel_gpu::op::SDPA::default_order(4));

        auto result = std::make_shared<ov::op::v0::Result>(sdpa);

        ov::ResultVector results{ result };

        model = std::make_shared<ov::Model>(results, params);
        manager.register_pass<KVCacheCompression>(ov::element::i8, false);
    }
    {
        ov::op::internal::DynamicQuantize::Attributes dq_config;
        dq_config.quantization_type = ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric;
        dq_config.quantization_dt = ov::element::i8;
        dq_config.scale_dt = ov::element::f16;
        dq_config.zp_dt = ov::element::f16;
        dq_config.group_sizes = { 1, 1, 1, UINT64_MAX };
        dq_config.scales_zp_output_order = { 0, 1, 2, 3 };
        dq_config.output_storage_type = ov::op::internal::DynamicQuantize::OutputStorageType::InterleavedScalesZP;

        auto query = std::make_shared<ov::op::v0::Parameter>(element_type, input_shape);
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});

        auto key_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v0"});
        auto key_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto key_past_variable_infos = { ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::i8, "v0"},
                                         ov::op::util::VariableInfo{{1, 32, -1, 2}, ov::element::f16, "v0"} };
        auto key_past_compressed = std::make_shared<ov::intel_gpu::op::ReadValues>(key_variable, key_past_variable_infos);
        auto key_cache_inputs = ov::OutputVector{ key_past_compressed->output(0), key_current, beam_idx, key_past_compressed->output(1) };
        auto key_cache = std::make_shared<ov::intel_gpu::op::KVCacheCompressed>(key_cache_inputs,
                                                                                key_variable,
                                                                                false,
                                                                                concat_axis,
                                                                                gather_axis,
                                                                                dq_config);

        auto value_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v1"});
        auto value_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto value_past_variable_infos = { ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::i8, "v1"},
                                           ov::op::util::VariableInfo{{1, 32, -1, 2}, ov::element::f16, "v1"} };
        auto value_past_compressed = std::make_shared<ov::intel_gpu::op::ReadValues>(value_variable, value_past_variable_infos);
        auto value_cache_inputs = ov::OutputVector{ value_past_compressed->output(0), value_current, beam_idx, value_past_compressed->output(1) };
        auto value_cache = std::make_shared<ov::intel_gpu::op::KVCacheCompressed>(value_cache_inputs,
                                                                                  value_variable,
                                                                                  false,
                                                                                  concat_axis,
                                                                                  gather_axis,
                                                                                  dq_config);

        ov::ParameterVector params{ beam_idx, query, key_current, value_current };

        if (with_mask) {
            auto attn_input = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape::dynamic(4));
            mask = attn_input;
            params.push_back(attn_input);
        }

        if (with_mask && with_scale) {
            auto scale_input = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape{1});
            scale = scale_input;
            params.push_back(scale_input);
        }

        ov::OutputVector sdpa_inputs = { query, key_cache->output(0), value_cache->output(0) };
        if (mask) {
            sdpa_inputs.push_back(mask);
        }

        if (scale) {
            sdpa_inputs.push_back(scale);
        }

        sdpa_inputs.push_back(key_cache->output(2));
        sdpa_inputs.push_back(value_cache->output(2));

        std::shared_ptr<ov::Node> sdpa = nullptr;
        sdpa = std::make_shared<ov::intel_gpu::op::IndirectSDPA>(sdpa_inputs,
                                                                 key_cache->output(1),
                                                                 causal,
                                                                 gather_axis,
                                                                 qkv_order,
                                                                 qkv_order,
                                                                 qkv_order,
                                                                 ov::intel_gpu::op::SDPA::default_order(4),
                                                                 dq_config);

        auto result = std::make_shared<ov::op::v0::Result>(sdpa);

        ov::ResultVector results{ result };

        model_ref = std::make_shared<ov::Model>(results, params);
    }
}

TEST_F(TransformationTestsF, KVCacheCompressionWithPastSeqLen) {
    bool causal = false;
    bool with_mask = true;
    bool with_scale = true;
    size_t concat_axis = 2;
    size_t gather_axis = 0;
    ov::element::Type_t element_type = ov::element::f16;
    std::vector<int64_t> qkv_order = {0, 1, 2, 3};
    std::shared_ptr<ov::Node> mask = nullptr;
    std::shared_ptr<ov::Node> scale = nullptr;
    ov::PartialShape input_shape = ov::PartialShape{1, 32, -1, 80};

    {
        auto query = std::make_shared<ov::op::v0::Parameter>(element_type, input_shape);
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});
        auto past_seq_len = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});

        auto key_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v0"});
        auto key_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto key_past = std::make_shared<ov::intel_gpu::op::ReadValue>(key_variable);
        auto key_cache = std::make_shared<ov::intel_gpu::op::KVCache>(key_past, key_current, beam_idx, past_seq_len, key_variable, concat_axis, gather_axis);

        auto value_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v1"});
        auto value_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto value_past = std::make_shared<ov::intel_gpu::op::ReadValue>(value_variable);
        auto value_cache =
            std::make_shared<ov::intel_gpu::op::KVCache>(value_past, value_current, beam_idx, past_seq_len, value_variable, concat_axis, gather_axis);

        ov::ParameterVector params{ past_seq_len, beam_idx, query, key_current, value_current };

        if (with_mask) {
            auto attn_mask = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape::dynamic(4));
            mask = attn_mask;
            params.push_back(attn_mask);
        }

        if (with_mask && with_scale) {
            auto scale_input = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape{1});
            scale = scale_input;
            params.push_back(scale_input);
        }

        ov::OutputVector sdpa_inputs = { query, key_cache->output(0), value_cache->output(0) };

        if (mask) {
            sdpa_inputs.push_back(mask);
        }

        if (scale) {
            sdpa_inputs.push_back(scale);
        }

        std::shared_ptr<ov::Node> sdpa = nullptr;
        sdpa = std::make_shared<ov::intel_gpu::op::IndirectSDPA>(sdpa_inputs,
                                                                 key_cache->output(1),
                                                                 causal,
                                                                 gather_axis,
                                                                 qkv_order,
                                                                 qkv_order,
                                                                 qkv_order,
                                                                 ov::intel_gpu::op::SDPA::default_order(4));

        auto result = std::make_shared<ov::op::v0::Result>(sdpa);

        ov::ResultVector results{ result };

        model = std::make_shared<ov::Model>(results, params);
        manager.register_pass<KVCacheCompression>(ov::element::i8, false);
    }
    {
        ov::op::internal::DynamicQuantize::Attributes dq_config;
        dq_config.quantization_type = ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric;
        dq_config.quantization_dt = ov::element::i8;
        dq_config.scale_dt = ov::element::f16;
        dq_config.zp_dt = ov::element::f16;
        dq_config.group_sizes = { 1, 1, 1, UINT64_MAX };
        dq_config.scales_zp_output_order = { 0, 1, 2, 3 };
        dq_config.output_storage_type = ov::op::internal::DynamicQuantize::OutputStorageType::InterleavedScalesZP;

        auto query = std::make_shared<ov::op::v0::Parameter>(element_type, input_shape);
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});
        auto past_seq_len = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});

        auto key_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v0"});
        auto key_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto key_past_variable_infos = { ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::i8, "v0"},
                                         ov::op::util::VariableInfo{{1, 32, -1, 2}, ov::element::f16, "v0"} };
        auto key_past_compressed = std::make_shared<ov::intel_gpu::op::ReadValues>(key_variable, key_past_variable_infos);
        auto key_cache_inputs = ov::OutputVector{ key_past_compressed->output(0), key_current, beam_idx, past_seq_len, key_past_compressed->output(1) };
        auto key_cache = std::make_shared<ov::intel_gpu::op::KVCacheCompressed>(key_cache_inputs,
                                                                                key_variable,
                                                                                true,
                                                                                concat_axis,
                                                                                gather_axis,
                                                                                dq_config);

        auto value_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v1"});
        auto value_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto value_past_variable_infos = { ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::i8, "v1"},
                                           ov::op::util::VariableInfo{{1, 32, -1, 2}, ov::element::f16, "v1"} };
        auto value_past_compressed = std::make_shared<ov::intel_gpu::op::ReadValues>(value_variable, value_past_variable_infos);
        auto value_cache_inputs = ov::OutputVector{value_past_compressed->output(0), value_current, beam_idx, past_seq_len, value_past_compressed->output(1)};
        auto value_cache = std::make_shared<ov::intel_gpu::op::KVCacheCompressed>(value_cache_inputs,
                                                                                  value_variable, 
                                                                                  true,
                                                                                  concat_axis,
                                                                                  gather_axis,
                                                                                  dq_config);

        ov::ParameterVector params{ past_seq_len, beam_idx, query, key_current, value_current };

        if (with_mask) {
            auto attn_input = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape::dynamic(4));
            mask = attn_input;
            params.push_back(attn_input);
        }

        if (with_mask && with_scale) {
            auto scale_input = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape{1});
            scale = scale_input;
            params.push_back(scale_input);
        }

        ov::OutputVector sdpa_inputs = { query, key_cache->output(0), value_cache->output(0) };
        if (mask) {
            sdpa_inputs.push_back(mask);
        }

        if (scale) {
            sdpa_inputs.push_back(scale);
        }

        sdpa_inputs.push_back(key_cache->output(2));
        sdpa_inputs.push_back(value_cache->output(2));

        std::shared_ptr<ov::Node> sdpa = nullptr;
        sdpa = std::make_shared<ov::intel_gpu::op::IndirectSDPA>(sdpa_inputs,
                                                                 key_cache->output(1),
                                                                 causal,
                                                                 gather_axis,
                                                                 qkv_order,
                                                                 qkv_order,
                                                                 qkv_order,
                                                                 ov::intel_gpu::op::SDPA::default_order(4),
                                                                 dq_config);

        auto result = std::make_shared<ov::op::v0::Result>(sdpa);

        ov::ResultVector results{ result };

        model_ref = std::make_shared<ov::Model>(results, params);
    }
}

TEST_F(TransformationTestsF, KVCacheCompressionWithInitializers) {
    bool causal = false;
    bool with_mask = true;
    bool with_scale = true;
    size_t concat_axis = 2;
    size_t gather_axis = 0;
    ov::element::Type_t element_type = ov::element::f16;
    std::vector<int64_t> qkv_order = {0, 1, 2, 3};
    std::shared_ptr<ov::Node> mask = nullptr;
    std::shared_ptr<ov::Node> scale = nullptr;
    ov::PartialShape input_shape = ov::PartialShape{1, 32, -1, 80};

    {
        auto query = std::make_shared<ov::op::v0::Parameter>(element_type, input_shape);
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});

        auto key_variable_initializer = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto key_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto key_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v0"});
        auto key_past = std::make_shared<ov::intel_gpu::op::ReadValue>(key_variable_initializer, key_variable);
        auto key_cache = std::make_shared<ov::intel_gpu::op::KVCache>(key_past, key_current, beam_idx, key_variable, concat_axis, gather_axis);

        auto value_variable_initializer = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto value_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v1"});
        auto value_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto value_past = std::make_shared<ov::intel_gpu::op::ReadValue>(value_variable_initializer, value_variable);
        auto value_cache = std::make_shared<ov::intel_gpu::op::KVCache>(value_past, value_current, beam_idx, value_variable, concat_axis, gather_axis);

        ov::ParameterVector params{ beam_idx, query, key_current, value_current, key_variable_initializer, value_variable_initializer };

        if (with_mask) {
            auto attn_mask = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape::dynamic(4));
            mask = attn_mask;
            params.push_back(attn_mask);
        }

        if (with_mask && with_scale) {
            auto scale_input = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape{1});
            scale = scale_input;
            params.push_back(scale_input);
        }

        ov::OutputVector sdpa_inputs = { query, key_cache->output(0), value_cache->output(0) };

        if (mask) {
            sdpa_inputs.push_back(mask);
        }

        if (scale) {
            sdpa_inputs.push_back(scale);
        }

        std::shared_ptr<ov::Node> sdpa = nullptr;
        sdpa = std::make_shared<ov::intel_gpu::op::IndirectSDPA>(sdpa_inputs,
                                                                 key_cache->output(1),
                                                                 causal,
                                                                 gather_axis,
                                                                 qkv_order,
                                                                 qkv_order,
                                                                 qkv_order,
                                                                 ov::intel_gpu::op::SDPA::default_order(4));

        auto result = std::make_shared<ov::op::v0::Result>(sdpa);

        ov::ResultVector results{ result };

        model = std::make_shared<ov::Model>(results, params);
        manager.register_pass<KVCacheCompression>(ov::element::i8, false);
    }
    {
        ov::op::internal::DynamicQuantize::Attributes dq_config;
        dq_config.quantization_type = ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric;
        dq_config.quantization_dt = ov::element::i8;
        dq_config.scale_dt = ov::element::f16;
        dq_config.zp_dt = ov::element::f16;
        dq_config.group_sizes = { 1, 1, 1, UINT64_MAX };
        dq_config.scales_zp_output_order = { 0, 1, 2, 3 };
        dq_config.output_storage_type = ov::op::internal::DynamicQuantize::OutputStorageType::InterleavedScalesZP;

        auto query = std::make_shared<ov::op::v0::Parameter>(element_type, input_shape);
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});

        auto key_past_variable_infos = { ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::i8, "v0"},
                                         ov::op::util::VariableInfo{{1, 32, -1, 2}, ov::element::f16, "v0"} };
        auto key_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto key_variable_initializer = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto key_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v0"});

        auto key_initializer_dq =
            std::make_shared<ov::op::internal::DynamicQuantize>(key_variable_initializer, dq_config);
        auto key_past_initializers = ov::OutputVector{ key_initializer_dq->output(0), key_initializer_dq->output(1) };
        auto key_past_compressed = std::make_shared<ov::intel_gpu::op::ReadValues>(key_past_initializers, key_variable, key_past_variable_infos);
        auto key_cache_inputs = ov::OutputVector{ key_past_compressed->output(0), key_current, beam_idx, key_past_compressed->output(1) };
        auto key_cache = std::make_shared<ov::intel_gpu::op::KVCacheCompressed>(key_cache_inputs,
                                                                                key_variable,
                                                                                false,
                                                                                concat_axis,
                                                                                gather_axis,
                                                                                dq_config);

        auto value_past_variable_infos = { ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::i8, "v1"},
                                           ov::op::util::VariableInfo{{1, 32, -1, 2}, ov::element::f16, "v1"} };

        auto value_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto value_variable_initializer = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto value_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v1"});

        auto value_initializer_dq =
            std::make_shared<ov::op::internal::DynamicQuantize>(value_variable_initializer, dq_config);
        auto value_past_initializers = ov::OutputVector{ value_initializer_dq->output(0), value_initializer_dq->output(1) };
        auto value_past_compressed = std::make_shared<ov::intel_gpu::op::ReadValues>(value_past_initializers, value_variable, value_past_variable_infos);
        auto value_cache_inputs = ov::OutputVector{ value_past_compressed->output(0), value_current, beam_idx, value_past_compressed->output(1) };
        auto value_cache = std::make_shared<ov::intel_gpu::op::KVCacheCompressed>(value_cache_inputs,
                                                                                  value_variable,
                                                                                  false,
                                                                                  concat_axis,
                                                                                  gather_axis,
                                                                                  dq_config);

        ov::ParameterVector params{ beam_idx, query, key_current, value_current, key_variable_initializer, value_variable_initializer };

        if (with_mask) {
            auto attn_input = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape::dynamic(4));
            mask = attn_input;
            params.push_back(attn_input);
        }

        if (with_mask && with_scale) {
            auto scale_input = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape{1});
            scale = scale_input;
            params.push_back(scale_input);
        }

        ov::OutputVector sdpa_inputs = { query, key_cache->output(0), value_cache->output(0) };
        if (mask) {
            sdpa_inputs.push_back(mask);
        }

        if (scale) {
            sdpa_inputs.push_back(scale);
        }

        sdpa_inputs.push_back(key_cache->output(2));
        sdpa_inputs.push_back(value_cache->output(2));

        std::shared_ptr<ov::Node> sdpa = nullptr;
        sdpa = std::make_shared<ov::intel_gpu::op::IndirectSDPA>(sdpa_inputs,
                                                                 key_cache->output(1),
                                                                 causal,
                                                                 gather_axis,
                                                                 qkv_order,
                                                                 qkv_order,
                                                                 qkv_order,
                                                                 ov::intel_gpu::op::SDPA::default_order(4),
                                                                 dq_config);

        auto result = std::make_shared<ov::op::v0::Result>(sdpa);

        ov::ResultVector results{ result };

        model_ref = std::make_shared<ov::Model>(results, params);
    }
}

TEST_F(TransformationTestsF, NewTestKVCacheCompression) {
    ov::PartialShape input_shape = ov::PartialShape{-1, 32, -1, 80};
    std::vector<int64_t> qkv_order = {0, 1, 2, 3};
    bool causal = false;
    size_t gather_axis = 0;
    size_t concat_axis = -2;
    {
        auto query = std::make_shared<ov::op::v0::Parameter>(element::f16, input_shape);
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});

        auto key_current_data = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 32, -1, 240});
        auto key_vs_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 3);
        auto key_vs_split_length = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, std::vector<int>{80, 80, -1});
        auto key_current_vs = std::make_shared<ov::op::v1::VariadicSplit>(key_current_data, key_vs_axis, key_vs_split_length);
        auto key_variable_initializer = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto key_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{-1, 32, -1, 80}, ov::element::f16, "v0"});
        auto key_past = std::make_shared<ov::intel_gpu::op::ReadValue>(key_variable_initializer, key_variable);
        auto key_cache = std::make_shared<ov::intel_gpu::op::KVCache>(key_past, key_current_vs->output(2), beam_idx, key_variable, concat_axis, gather_axis);

        auto value_current_data = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 32, -1, 240});
        auto value_vs_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 3);
        auto value_vs_split_length = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, std::vector<int>{80, 80, -1});
        auto value_current_vs = std::make_shared<ov::op::v1::VariadicSplit>(value_current_data, value_vs_axis, value_vs_split_length);
        auto value_variable_initializer = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto value_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{-1, 32, -1, 80}, ov::element::f16, "v1"});
        auto value_past = std::make_shared<ov::intel_gpu::op::ReadValue>(value_variable_initializer, value_variable);
        auto value_cache = std::make_shared<ov::intel_gpu::op::KVCache>(value_past, value_current_vs->output(2), beam_idx, value_variable, concat_axis, gather_axis);

        auto attn_mask = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 1, -1, -1});
        auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{}, 0.111816f);
        auto sdpa = std::make_shared<ov::intel_gpu::op::IndirectSDPA>(ov::OutputVector{query, key_cache->output(0), value_cache->output(0), attn_mask, scale},
                                                                      key_cache->output(1),
                                                                      causal,
                                                                      gather_axis,
                                                                      qkv_order,
                                                                      qkv_order,
                                                                      qkv_order,
                                                                      ov::intel_gpu::op::SDPA::default_order(4));
        auto result = std::make_shared<ov::op::v0::Result>(sdpa);

        auto params = ParameterVector{query, beam_idx, key_variable_initializer, key_current_data, value_variable_initializer, value_current_data, attn_mask};
        model = std::make_shared<ov::Model>(ResultVector{result}, params);
        manager.register_pass<KVCacheCompression>(ov::element::i8, false);
    }
    {
        ov::op::internal::DynamicQuantize::Attributes dq_config;
        dq_config.quantization_type = ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric;
        dq_config.quantization_dt = ov::element::i8;
        dq_config.scale_dt = ov::element::f16;
        dq_config.zp_dt = ov::element::f16;
        dq_config.group_sizes = { 1, 1, 1, UINT64_MAX };
        dq_config.scales_zp_output_order = { 0, 1, 2, 3 };
        dq_config.output_storage_type = ov::op::internal::DynamicQuantize::OutputStorageType::InterleavedScalesZP;

        auto query = std::make_shared<ov::op::v0::Parameter>(element::f16, input_shape);
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});

        auto key_past_variable_infos = { ov::op::util::VariableInfo{{-1, 32, -1, 80}, ov::element::i8, "v0"},
                                         ov::op::util::VariableInfo{{-1, 32, -1, 2}, ov::element::f16, "v0"} };

        auto key_current_data = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 32, -1, 240});
        auto key_vs_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 3);
        auto key_vs_split_length = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, std::vector<int>{80, 80, -1});
        auto key_current_vs = std::make_shared<ov::op::v1::VariadicSplit>(key_current_data, key_vs_axis, key_vs_split_length);
        auto key_variable_initializer = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto key_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{-1, 32, -1, 80}, ov::element::f16, "v0"});

        auto key_initializer_dq =
            std::make_shared<ov::op::internal::DynamicQuantize>(key_variable_initializer, dq_config);
        auto key_past_initializers = ov::OutputVector{key_initializer_dq->output(0), key_initializer_dq->output(1)};
        auto key_past_compressed = std::make_shared<ov::intel_gpu::op::ReadValues>(key_past_initializers, key_variable, key_past_variable_infos);
        auto key_cache_inputs = ov::OutputVector{key_past_compressed->output(0), key_current_vs->output(2), beam_idx, key_past_compressed->output(1)};
        auto key_cache = std::make_shared<ov::intel_gpu::op::KVCacheCompressed>(key_cache_inputs,
                                                                                key_variable,
                                                                                false,
                                                                                concat_axis,
                                                                                gather_axis,
                                                                                dq_config);

        auto value_past_variable_infos = {ov::op::util::VariableInfo{{-1, 32, -1, 80}, ov::element::i8, "v1"},
                                          ov::op::util::VariableInfo{{-1, 32, -1, 2}, ov::element::f16, "v1"}};

        auto value_current_data = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 32, -1, 240});
        auto value_vs_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 3);
        auto value_vs_split_length = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, std::vector<int>{80, 80, -1});
        auto value_current_vs = std::make_shared<ov::op::v1::VariadicSplit>(value_current_data, value_vs_axis, value_vs_split_length);
        auto value_variable_initializer = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto value_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v1"});

        auto value_initializer_dq =
            std::make_shared<ov::op::internal::DynamicQuantize>(value_variable_initializer, dq_config);
        auto value_past_initializers = ov::OutputVector{value_initializer_dq->output(0), value_initializer_dq->output(1)};
        auto value_past_compressed = std::make_shared<ov::intel_gpu::op::ReadValues>(value_past_initializers, value_variable, value_past_variable_infos);
        auto value_cache_inputs = ov::OutputVector{value_past_compressed->output(0), value_current_vs->output(2), beam_idx, value_past_compressed->output(1)};
        auto value_cache = std::make_shared<ov::intel_gpu::op::KVCacheCompressed>(value_cache_inputs,
                                                                                  value_variable,
                                                                                  false,
                                                                                  concat_axis,
                                                                                  gather_axis,
                                                                                  dq_config);

        auto attn_mask = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 1, -1, -1});
        auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{}, 0.111816f);
        auto sdpa = std::make_shared<ov::intel_gpu::op::IndirectSDPA>(ov::OutputVector{query, key_cache->output(0), value_cache->output(0),
                                                                                       attn_mask, scale,
                                                                                       key_cache->output(2), value_cache->output(2)},
                                                                      key_cache->output(1),
                                                                      causal,
                                                                      gather_axis,
                                                                      qkv_order,
                                                                      qkv_order,
                                                                      qkv_order,
                                                                      ov::intel_gpu::op::SDPA::default_order(4),
                                                                      dq_config);
        auto result = std::make_shared<ov::op::v0::Result>(sdpa);

        auto params = ParameterVector{query, beam_idx, key_variable_initializer, key_current_data, value_variable_initializer, value_current_data, attn_mask};
        model_ref = std::make_shared<ov::Model>(ResultVector{result}, params);
    }
}

// Test KV cache compression with Planar output storage (supports_immad=true),
// where scales and zero points are stored as separate buffers.
// This exercises the code path fixed for separate key_zp/value_zp kernel arguments in sdpa_opt.cl.
TEST_F(TransformationTestsF, KVCacheCompressionPlanar) {
    bool causal = false;
    bool with_mask = true;
    bool with_scale = true;
    size_t concat_axis = 2;
    size_t gather_axis = 0;
    ov::element::Type_t element_type = ov::element::f16;
    std::vector<int64_t> qkv_order = {0, 1, 2, 3};
    std::shared_ptr<ov::Node> mask = nullptr;
    std::shared_ptr<ov::Node> scale = nullptr;
    ov::PartialShape input_shape = ov::PartialShape{1, 32, -1, 80};

    {
        auto query = std::make_shared<ov::op::v0::Parameter>(element_type, input_shape);
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});

        auto key_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v0"});
        auto key_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto key_past = std::make_shared<ov::intel_gpu::op::ReadValue>(key_variable);
        auto key_cache = std::make_shared<ov::intel_gpu::op::KVCache>(key_past, key_current, beam_idx, key_variable, concat_axis, gather_axis);

        auto value_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v1"});
        auto value_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto value_past = std::make_shared<ov::intel_gpu::op::ReadValue>(value_variable);
        auto value_cache = std::make_shared<ov::intel_gpu::op::KVCache>(value_past, value_current, beam_idx, value_variable, concat_axis, gather_axis);

        ov::ParameterVector params{ beam_idx, query, key_current, value_current };

        if (with_mask) {
            auto attn_mask = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape::dynamic(4));
            mask = attn_mask;
            params.push_back(attn_mask);
        }

        if (with_mask && with_scale) {
            auto scale_input = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape{1});
            scale = scale_input;
            params.push_back(scale_input);
        }

        ov::OutputVector sdpa_inputs = { query, key_cache->output(0), value_cache->output(0) };

        if (mask) {
            sdpa_inputs.push_back(mask);
        }

        if (scale) {
            sdpa_inputs.push_back(scale);
        }

        std::shared_ptr<ov::Node> sdpa = nullptr;
        sdpa = std::make_shared<ov::intel_gpu::op::IndirectSDPA>(sdpa_inputs,
                                                                 key_cache->output(1),
                                                                 causal,
                                                                 gather_axis,
                                                                 qkv_order,
                                                                 qkv_order,
                                                                 qkv_order,
                                                                 ov::intel_gpu::op::SDPA::default_order(4));

        auto result = std::make_shared<ov::op::v0::Result>(sdpa);

        ov::ResultVector results{ result };

        model = std::make_shared<ov::Model>(results, params);
        // supports_immad=true triggers Planar output storage with separate ZP buffers
        manager.register_pass<KVCacheCompression>(ov::element::i8, true);
    }
    {
        ov::op::internal::DynamicQuantize::Attributes dq_config;
        dq_config.quantization_type = ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric;
        dq_config.quantization_dt = ov::element::i8;
        dq_config.scale_dt = ov::element::f16;
        dq_config.zp_dt = ov::element::i8;
        dq_config.group_sizes = { 1, 1, 1, UINT64_MAX };
        dq_config.scales_zp_output_order = { 0, 1, 2, 3 };
        dq_config.output_storage_type = ov::op::internal::DynamicQuantize::OutputStorageType::Planar;

        auto query = std::make_shared<ov::op::v0::Parameter>(element_type, input_shape);
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});

        // For Planar mode, ReadValues produces 3 outputs: data(i8), scale(f16), zp(i8)
        auto key_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v0"});
        auto key_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto key_past_variable_infos = { ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::i8, "v0"},
                                         ov::op::util::VariableInfo{{1, 32, -1, 1}, ov::element::f16, "v0"},
                                         ov::op::util::VariableInfo{{1, 32, -1, 1}, ov::element::i8, "v0"} };
        auto key_past_compressed = std::make_shared<ov::intel_gpu::op::ReadValues>(key_variable, key_past_variable_infos);
        auto key_cache_inputs = ov::OutputVector{ key_past_compressed->output(0), key_current, beam_idx,
                                                  key_past_compressed->output(1), key_past_compressed->output(2) };
        auto key_cache = std::make_shared<ov::intel_gpu::op::KVCacheCompressed>(key_cache_inputs,
                                                                                key_variable,
                                                                                false,
                                                                                concat_axis,
                                                                                gather_axis,
                                                                                dq_config);

        auto value_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v1"});
        auto value_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto value_past_variable_infos = { ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::i8, "v1"},
                                           ov::op::util::VariableInfo{{1, 32, -1, 1}, ov::element::f16, "v1"},
                                           ov::op::util::VariableInfo{{1, 32, -1, 1}, ov::element::i8, "v1"} };
        auto value_past_compressed = std::make_shared<ov::intel_gpu::op::ReadValues>(value_variable, value_past_variable_infos);
        auto value_cache_inputs = ov::OutputVector{ value_past_compressed->output(0), value_current, beam_idx,
                                                    value_past_compressed->output(1), value_past_compressed->output(2) };
        auto value_cache = std::make_shared<ov::intel_gpu::op::KVCacheCompressed>(value_cache_inputs,
                                                                                  value_variable,
                                                                                  false,
                                                                                  concat_axis,
                                                                                  gather_axis,
                                                                                  dq_config);

        ov::ParameterVector params{ beam_idx, query, key_current, value_current };

        if (with_mask) {
            auto attn_input = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape::dynamic(4));
            mask = attn_input;
            params.push_back(attn_input);
        }

        if (with_mask && with_scale) {
            auto scale_input = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape{1});
            scale = scale_input;
            params.push_back(scale_input);
        }

        ov::OutputVector sdpa_inputs = { query, key_cache->output(0), value_cache->output(0) };
        if (mask) {
            sdpa_inputs.push_back(mask);
        }

        if (scale) {
            sdpa_inputs.push_back(scale);
        }

        // For Planar mode: key_scale, value_scale, key_zp, value_zp
        sdpa_inputs.push_back(key_cache->output(2));
        sdpa_inputs.push_back(value_cache->output(2));
        sdpa_inputs.push_back(key_cache->output(3));
        sdpa_inputs.push_back(value_cache->output(3));

        std::shared_ptr<ov::Node> sdpa = nullptr;
        sdpa = std::make_shared<ov::intel_gpu::op::IndirectSDPA>(sdpa_inputs,
                                                                 key_cache->output(1),
                                                                 causal,
                                                                 gather_axis,
                                                                 qkv_order,
                                                                 qkv_order,
                                                                 qkv_order,
                                                                 ov::intel_gpu::op::SDPA::default_order(4),
                                                                 dq_config);

        auto result = std::make_shared<ov::op::v0::Result>(sdpa);

        ov::ResultVector results{ result };

        model_ref = std::make_shared<ov::Model>(results, params);
    }
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
