// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>

#include "intel_npu/config/config.hpp"
#include "intel_npu/config/npuw.hpp"
#include "openvino/op/ops.hpp"
#include "partitioning/online/compiler.hpp"
#include "partitioning/patterns/sdpa.hpp"

namespace {

::intel_npu::Config make_cfg(const ::intel_npu::Config::ConfigMap& cfg_map) {
    auto opt_desc = std::make_shared<::intel_npu::OptionsDesc>();
    ::intel_npu::registerNPUWOptions(*opt_desc);
    auto cfg = ::intel_npu::Config(opt_desc);
    cfg.update(cfg_map);
    return cfg;
}

// Counts how many groups have a specific isolation tag in the partitioning result.
size_t count_groups_with_tag(const ov::npuw::Ensemble& ens, const std::string& tag) {
    return std::count_if(ens.groups.begin(), ens.groups.end(), [&tag](const ov::npuw::Group& group) {
        return group.gettag() == tag;
    });
}

// Build a minimal model with decomposed SDPA pattern (no ScaledDotProductAttention op).
// Graph structure per layer:
//   past_key(f16) → Convert(f32) → Concat(new_key) → [opt:Unsqueeze→Broadcast→Reshape] → MatMul(query) → Add(mask) → Softmax → MatMul(value_concat) → Transpose → Reshape → Result
//   past_value(f16) → Convert(f32) → Concat(new_value) → [opt:Unsqueeze→Broadcast→Reshape] →                                                          ↑
//
// Parameters:
//   num_layers: number of repeated attention layers
//   with_gqa: if true, add Unsqueeze → Broadcast → Reshape for GQA expansion
std::shared_ptr<ov::Model> build_decomposed_sdpa_model(size_t num_layers = 1,
                                                       bool with_gqa = false,
                                                       size_t num_heads = 4,
                                                       size_t num_kv_heads = 4,
                                                       size_t head_dim = 16,
                                                       size_t query_len = 8,
                                                       size_t past_len = 8) {
    using namespace ov;

    const size_t context_len = past_len + query_len;
    const Shape query_shape = {1, num_heads, query_len, head_dim};
    const Shape past_shape = {1, num_kv_heads, past_len, head_dim};
    const Shape new_token_shape = {1, num_kv_heads, query_len, head_dim};
    const Shape mask_shape = {1, 1, query_len, context_len};

    ParameterVector params;
    ResultVector results;

    for (size_t n = 0; n < num_layers; ++n) {
        const std::string idx = std::to_string(n);
        auto make_param = [&](const std::string& name, const Shape& shape, element::Type et = element::f16) {
            auto p = std::make_shared<op::v0::Parameter>(et, shape);
            p->set_friendly_name(name);
            p->output(0).get_tensor().set_names({name});
            params.push_back(p);
            return p;
        };

        auto query = make_param("query." + idx, query_shape, element::f32);
        auto past_key = make_param("past_key_values." + idx + ".key", past_shape);
        auto past_value = make_param("past_key_values." + idx + ".value", past_shape);
        auto new_key = make_param("new_key." + idx, new_token_shape, element::f32);
        auto new_value = make_param("new_value." + idx, new_token_shape, element::f32);
        auto mask = make_param("mask." + idx, mask_shape, element::f32);

        // Convert(f16 → f32) before Concat — this is what PPP inserts and the pattern matches
        auto cvt_key = std::make_shared<op::v0::Convert>(past_key, element::f32);
        cvt_key->set_friendly_name("convert_key." + idx);
        auto cvt_value = std::make_shared<op::v0::Convert>(past_value, element::f32);
        cvt_value->set_friendly_name("convert_value." + idx);

        auto key_concat = std::make_shared<op::v0::Concat>(OutputVector{cvt_key, new_key}, 2);
        key_concat->set_friendly_name("concat_key." + idx);
        auto value_concat = std::make_shared<op::v0::Concat>(OutputVector{cvt_value, new_value}, 2);
        value_concat->set_friendly_name("concat_value." + idx);

        Output<Node> key_for_matmul = key_concat->output(0);
        Output<Node> value_for_matmul = value_concat->output(0);

        // GQA expansion: Unsqueeze → Broadcast → Reshape
        if (with_gqa && num_kv_heads < num_heads) {
            const size_t groups = num_heads / num_kv_heads;
            auto expand_gqa = [&](Output<Node> input, const std::string& kv) -> Output<Node> {
                auto unsq = std::make_shared<op::v0::Unsqueeze>(
                    input,
                    op::v0::Constant::create(element::i64, Shape{1}, {2}));
                unsq->set_friendly_name("unsqueeze_" + kv + "." + idx);

                auto bcast = std::make_shared<op::v3::Broadcast>(
                    unsq,
                    op::v0::Constant::create(element::i64, Shape{5},
                                             std::vector<int64_t>{1, static_cast<int64_t>(num_kv_heads),
                                                                  static_cast<int64_t>(groups),
                                                                  static_cast<int64_t>(context_len),
                                                                  static_cast<int64_t>(head_dim)}));
                bcast->set_friendly_name("broadcast_" + kv + "." + idx);

                auto rshp = std::make_shared<op::v1::Reshape>(
                    bcast,
                    op::v0::Constant::create(element::i64, Shape{4},
                                             std::vector<int64_t>{1, static_cast<int64_t>(num_heads),
                                                                  static_cast<int64_t>(context_len),
                                                                  static_cast<int64_t>(head_dim)}),
                    false);
                rshp->set_friendly_name("reshape_" + kv + "." + idx);
                return rshp->output(0);
            };
            key_for_matmul = expand_gqa(key_concat->output(0), "key");
            value_for_matmul = expand_gqa(value_concat->output(0), "value");
        }

        // Q @ K^T → Add(mask) → Softmax → @ V → Transpose → Reshape
        auto qk = std::make_shared<op::v0::MatMul>(query, key_for_matmul, false, true);
        qk->set_friendly_name("matmul_qk." + idx);

        auto add = std::make_shared<op::v1::Add>(qk, mask);
        add->set_friendly_name("add_mask." + idx);

        auto softmax = std::make_shared<op::v8::Softmax>(add, -1);
        softmax->set_friendly_name("softmax." + idx);

        auto sv = std::make_shared<op::v0::MatMul>(softmax, value_for_matmul);
        sv->set_friendly_name("matmul_sv." + idx);

        auto transpose = std::make_shared<op::v1::Transpose>(
            sv, op::v0::Constant::create(element::i64, Shape{4}, std::vector<int64_t>{0, 2, 1, 3}));
        transpose->set_friendly_name("transpose." + idx);

        auto reshape = std::make_shared<op::v1::Reshape>(
            transpose,
            op::v0::Constant::create(element::i64, Shape{3},
                                     std::vector<int64_t>{1, static_cast<int64_t>(query_len),
                                                          static_cast<int64_t>(num_heads * head_dim)}),
            false);
        reshape->set_friendly_name("reshape." + idx);

        auto make_result = [&](const Output<Node>& out, const std::string& name) {
            auto r = std::make_shared<op::v0::Result>(out);
            r->set_friendly_name(name);
            results.push_back(r);
        };
        make_result(key_concat->output(0), "present." + idx + ".key");
        make_result(value_concat->output(0), "present." + idx + ".value");
        make_result(reshape->output(0), "attn_out." + idx);
    }

    auto model = std::make_shared<Model>(results, params, "decomposed_sdpa_model");
    model->validate_nodes_and_infer_types();
    return model;
}

// Build a model with decomposed SDPA pattern where past KV inputs have dynamic
// dequantization nodes (Subtract + Multiply) between Convert and Concat.
// This is what ConvertKVCacheToPrecision(i8) produces:
//   past_key(i8) → Convert(f32) → Subtract(Convert(zp_param)) → Multiply(scale_param) → Concat → ...
std::shared_ptr<ov::Model> build_decomposed_sdpa_dq_model(size_t num_layers = 1,
                                                          size_t num_heads = 4,
                                                          size_t head_dim = 16,
                                                          size_t query_len = 8,
                                                          size_t past_len = 8) {
    using namespace ov;

    const size_t context_len = past_len + query_len;
    const Shape query_shape = {1, num_heads, query_len, head_dim};
    const Shape past_shape = {1, num_heads, past_len, head_dim};
    const Shape new_token_shape = {1, num_heads, query_len, head_dim};
    const Shape mask_shape = {1, 1, query_len, context_len};
    // Scale/zp shapes: head_dim axis set to 1 (per-token quantization)
    const Shape key_scale_shape = {1, num_heads, past_len, 1};
    const Shape value_scale_shape = {1, num_heads, 1, head_dim};  // value reduces along dim 2

    ParameterVector params;
    ResultVector results;

    for (size_t n = 0; n < num_layers; ++n) {
        const std::string idx = std::to_string(n);
        auto make_param = [&](const std::string& name, const Shape& shape, element::Type et) {
            auto p = std::make_shared<op::v0::Parameter>(et, shape);
            p->set_friendly_name(name);
            p->output(0).get_tensor().set_names({name});
            params.push_back(p);
            return p;
        };

        auto query = make_param("query." + idx, query_shape, element::f32);
        auto past_key = make_param("past_key_values." + idx + ".key", past_shape, element::i8);
        auto past_value = make_param("past_key_values." + idx + ".value", past_shape, element::i8);
        auto new_key = make_param("new_key." + idx, new_token_shape, element::f32);
        auto new_value = make_param("new_value." + idx, new_token_shape, element::f32);
        auto mask = make_param("mask." + idx, mask_shape, element::f32);

        // DQ scale and zp parameters
        auto key_scale = make_param("DynamicQuantize/" + idx + "/past_key_values/key/scale", key_scale_shape,
                                    element::f32);
        auto key_zp =
            make_param("DynamicQuantize/" + idx + "/past_key_values/key/zp", key_scale_shape, element::i8);
        auto value_scale = make_param("DynamicQuantize/" + idx + "/past_key_values/value/scale", value_scale_shape,
                                      element::f32);

        // Past key path: Convert(i8→f32) → Subtract(Convert(zp)) → Multiply(scale) → Concat
        auto cvt_key = std::make_shared<op::v0::Convert>(past_key, element::f32);
        cvt_key->set_friendly_name("convert_key." + idx);

        auto cvt_key_zp = std::make_shared<op::v0::Convert>(key_zp, element::f32);
        cvt_key_zp->set_friendly_name("convert_key_zp." + idx);

        auto key_sub = std::make_shared<op::v1::Subtract>(cvt_key, cvt_key_zp);
        key_sub->set_friendly_name("subtract_key." + idx);

        auto key_mul = std::make_shared<op::v1::Multiply>(key_sub, key_scale);
        key_mul->set_friendly_name("multiply_key." + idx);

        auto key_concat = std::make_shared<op::v0::Concat>(OutputVector{key_mul, new_key}, 2);
        key_concat->set_friendly_name("concat_key." + idx);

        // Past value path: Convert(i8→f32) → Multiply(scale) → Concat (symmetric, no zp)
        auto cvt_value = std::make_shared<op::v0::Convert>(past_value, element::f32);
        cvt_value->set_friendly_name("convert_value." + idx);

        auto value_mul = std::make_shared<op::v1::Multiply>(cvt_value, value_scale);
        value_mul->set_friendly_name("multiply_value." + idx);

        auto value_concat = std::make_shared<op::v0::Concat>(OutputVector{value_mul, new_value}, 2);
        value_concat->set_friendly_name("concat_value." + idx);

        // Q @ K^T → Add(mask) → Softmax → @ V → Transpose → Reshape
        auto qk = std::make_shared<op::v0::MatMul>(query, key_concat, false, true);
        qk->set_friendly_name("matmul_qk." + idx);

        auto add = std::make_shared<op::v1::Add>(qk, mask);
        add->set_friendly_name("add_mask." + idx);

        auto softmax = std::make_shared<op::v8::Softmax>(add, -1);
        softmax->set_friendly_name("softmax." + idx);

        auto sv = std::make_shared<op::v0::MatMul>(softmax, value_concat);
        sv->set_friendly_name("matmul_sv." + idx);

        auto transpose = std::make_shared<op::v1::Transpose>(
            sv, op::v0::Constant::create(element::i64, Shape{4}, std::vector<int64_t>{0, 2, 1, 3}));
        transpose->set_friendly_name("transpose." + idx);

        auto reshape = std::make_shared<op::v1::Reshape>(
            transpose,
            op::v0::Constant::create(element::i64, Shape{3},
                                     std::vector<int64_t>{1, static_cast<int64_t>(query_len),
                                                          static_cast<int64_t>(num_heads * head_dim)}),
            false);
        reshape->set_friendly_name("reshape." + idx);

        auto make_result = [&](const Output<Node>& out, const std::string& name) {
            auto r = std::make_shared<op::v0::Result>(out);
            r->set_friendly_name(name);
            results.push_back(r);
        };
        make_result(key_concat->output(0), "present." + idx + ".key");
        make_result(value_concat->output(0), "present." + idx + ".value");
        make_result(reshape->output(0), "attn_out." + idx);
    }

    auto model = std::make_shared<Model>(results, params, "decomposed_sdpa_dq_model");
    model->validate_nodes_and_infer_types();
    return model;
}

}  // namespace

// --- SDPADecomposed matcher tests ---

TEST(SDPAPatternMatcherTest, SDPADecomposedMatchesSingleLayerModel) {
    auto model = build_decomposed_sdpa_model(/*num_layers=*/1);
    auto cfg = make_cfg({{"NPUW_ONLINE_PIPELINE", "REP"}, {"NPUW_ONLINE_ISOLATE", "P:SDPADecomposed/attn"}});
    auto ens = ov::npuw::online::buildPartitioning(model, cfg);

    EXPECT_GE(count_groups_with_tag(ens, "attn"), 1u)
        << "SDPADecomposed pattern should match the decomposed attention subgraph";
}

TEST(SDPAPatternMatcherTest, SDPADecomposedMatchesMultiLayerModel) {
    auto model = build_decomposed_sdpa_model(/*num_layers=*/4);
    auto cfg = make_cfg({{"NPUW_ONLINE_PIPELINE", "REP"}, {"NPUW_ONLINE_ISOLATE", "P:SDPADecomposed/attn"}});
    auto ens = ov::npuw::online::buildPartitioning(model, cfg);

    const auto attn_count = count_groups_with_tag(ens, "attn");
    EXPECT_GE(attn_count, 4u) << "SDPADecomposed should match all 4 attention layers, got " << attn_count;
}

TEST(SDPAPatternMatcherTest, SDPADecomposedDoesNotMatchDQModel) {
    // The existing SDPADecomposed pattern expects Convert → Concat.
    // After i8 DQ, the chain is Convert → Subtract → Multiply → Concat — no match expected.
    auto model = build_decomposed_sdpa_dq_model(/*num_layers=*/1);
    auto cfg = make_cfg({{"NPUW_ONLINE_PIPELINE", "REP"}, {"NPUW_ONLINE_ISOLATE", "P:SDPADecomposed/attn"}});
    auto ens = ov::npuw::online::buildPartitioning(model, cfg);

    EXPECT_EQ(count_groups_with_tag(ens, "attn"), 0u)
        << "SDPADecomposed should NOT match i8 DQ model (Convert → Subtract → Multiply → Concat)";
}

// --- SDPACompressed matcher tests ---

TEST(SDPAPatternMatcherTest, SDPACompressedMatchesSingleLayerDQModel) {
    auto model = build_decomposed_sdpa_dq_model(/*num_layers=*/1);
    auto cfg = make_cfg({{"NPUW_ONLINE_PIPELINE", "REP"}, {"NPUW_ONLINE_ISOLATE", "P:SDPACompressed/attn"}});
    auto ens = ov::npuw::online::buildPartitioning(model, cfg);

    EXPECT_GE(count_groups_with_tag(ens, "attn"), 1u)
        << "SDPACompressed pattern should match the i8 DQ attention subgraph";
}

TEST(SDPAPatternMatcherTest, SDPACompressedMatchesMultiLayerDQModel) {
    auto model = build_decomposed_sdpa_dq_model(/*num_layers=*/4);
    auto cfg = make_cfg({{"NPUW_ONLINE_PIPELINE", "REP"}, {"NPUW_ONLINE_ISOLATE", "P:SDPACompressed/attn"}});
    auto ens = ov::npuw::online::buildPartitioning(model, cfg);

    const auto attn_count = count_groups_with_tag(ens, "attn");
    EXPECT_GE(attn_count, 4u) << "SDPACompressed should match all 4 DQ attention layers, got " << attn_count;
}

TEST(SDPAPatternMatcherTest, SDPACompressedDoesNotMatchNonDQModel) {
    // SDPACompressed should not match a non-DQ model (Convert → Concat without Subtract/Multiply)
    auto model = build_decomposed_sdpa_model(/*num_layers=*/1);
    auto cfg = make_cfg({{"NPUW_ONLINE_PIPELINE", "REP"}, {"NPUW_ONLINE_ISOLATE", "P:SDPACompressed/attn"}});
    auto ens = ov::npuw::online::buildPartitioning(model, cfg);

    EXPECT_EQ(count_groups_with_tag(ens, "attn"), 0u)
        << "SDPACompressed should NOT match non-DQ model";
}

// --- ATTN preset test (boxed: runs all SDPA matchers via the ATTN preset) ---

TEST(SDPAPatternMatcherTest, ATTNPresetMatchesDecomposedModel) {
    auto model = build_decomposed_sdpa_model(/*num_layers=*/2);
    auto cfg = make_cfg({{"NPUW_ONLINE_PIPELINE", "REP"}, {"NPUW_ONLINE_ISOLATE", "ATTN"}});
    auto ens = ov::npuw::online::buildPartitioning(model, cfg);

    EXPECT_GE(count_groups_with_tag(ens, "attn"), 2u)
        << "ATTN preset should match decomposed SDPA model via SDPADecomposed pattern";
}

TEST(SDPAPatternMatcherTest, ATTNPresetMatchesDQModel) {
    auto model = build_decomposed_sdpa_dq_model(/*num_layers=*/2);
    auto cfg = make_cfg({{"NPUW_ONLINE_PIPELINE", "REP"}, {"NPUW_ONLINE_ISOLATE", "ATTN"}});
    auto ens = ov::npuw::online::buildPartitioning(model, cfg);

    EXPECT_GE(count_groups_with_tag(ens, "attn"), 2u)
        << "ATTN preset should match DQ SDPA model via SDPACompressed pattern";
}
