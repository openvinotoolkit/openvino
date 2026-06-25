// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "intel_npu/config/config.hpp"
#include "intel_npu/config/npuw.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/transpose.hpp"
#include "partitioning/online/compiler.hpp"
#include "partitioning/partitioning.hpp"

namespace {

::intel_npu::Config make_cfg(const ::intel_npu::Config::ConfigMap& cfg_map) {
    auto opt_desc = std::make_shared<::intel_npu::OptionsDesc>();
    ::intel_npu::registerNPUWOptions(*opt_desc);
    auto cfg = ::intel_npu::Config(opt_desc);
    cfg.update(cfg_map);
    return cfg;
}

size_t count_groups_with_tag(const ov::npuw::Ensemble& ens, const std::string& tag) {
    return static_cast<size_t>(std::count_if(ens.groups.begin(), ens.groups.end(), [&tag](const ov::npuw::Group& g) {
        return g.gettag() == tag;
    }));
}

// Build a decomposed SDPA model with Transpose+Reshape tail so it matches the
// SDPADecomposed pattern (whose root is Reshape after Transpose after MatMul2).
//
// Topology per layer:
//   past_key/value [params] -> Concat (axis 2)
//   query -> MatMul(query, key_concat^T)
//         -> Add(+mask) -> Softmax -> MatMul(softmax, value_concat)
//         -> Transpose([0,2,1,3]) -> Reshape([1, query_len, heads*head_dim])
//
// With num_blocks > 0: past_key_block_0, ..., past_key_block_{N-1} are used
//   instead of a single past_key param (no Convert nodes, direct block params).
std::shared_ptr<ov::Model> build_decomposed_sdpa_model(size_t num_layers,
                                                       size_t num_key_blocks = 0,
                                                       size_t block_size = 4) {
    using namespace ov;

    const size_t num_heads = 4;
    const size_t head_dim = 16;
    const size_t query_len = 1;

    // For contiguous: past_len drives context; for block: block*block_size tokens
    const size_t past_len = (num_key_blocks == 0) ? 63u : num_key_blocks * block_size;
    const size_t total_kv_len = past_len + query_len;

    const Shape past_shape = {1, num_heads, past_len, head_dim};
    const Shape block_shape = {1, num_heads, block_size, head_dim};
    const Shape new_token_shape = {1, num_heads, query_len, head_dim};
    const Shape mask_shape = {1, 1, query_len, total_kv_len};

    ParameterVector params;
    ResultVector results;

    for (size_t n = 0; n < num_layers; ++n) {
        const std::string idx = std::to_string(n);

        auto make_param = [&](const std::string& name, const Shape& shape) {
            auto p = std::make_shared<op::v0::Parameter>(element::f32, shape);
            p->set_friendly_name(name);
            p->output(0).get_tensor().set_names({name});
            params.push_back(p);
            return p;
        };

        auto query = make_param("query." + idx, new_token_shape);
        auto new_key = make_param("new_key." + idx, new_token_shape);
        auto new_value = make_param("new_value." + idx, new_token_shape);
        auto mask = make_param("mask." + idx, mask_shape);

        // Build KV Concat inputs
        OutputVector key_inputs, value_inputs;
        if (num_key_blocks == 0) {
            // Standard contiguous past KV
            key_inputs.push_back(make_param("past_key_values." + idx + ".key", past_shape)->output(0));
            value_inputs.push_back(make_param("past_key_values." + idx + ".value", past_shape)->output(0));
        } else {
            // Block-split KV cache: multiple block params, no Convert wrappers.
            // This is the structure produced by SplitKVCacheIntoBlocks and is the
            // primary scenario covered by our SDPADecomposed Concat change.
            for (size_t b = 0; b < num_key_blocks; ++b) {
                const std::string bsuf = "_block_" + std::to_string(b);
                key_inputs.push_back(make_param("past_key_values." + idx + ".key" + bsuf, block_shape)->output(0));
                value_inputs.push_back(make_param("past_key_values." + idx + ".value" + bsuf, block_shape)->output(0));
            }
        }
        key_inputs.push_back(new_key->output(0));
        value_inputs.push_back(new_value->output(0));

        auto key_concat = std::make_shared<op::v0::Concat>(key_inputs, 2);
        key_concat->set_friendly_name("concat_key." + idx);
        auto value_concat = std::make_shared<op::v0::Concat>(value_inputs, 2);
        value_concat->set_friendly_name("concat_value." + idx);

        auto qk = std::make_shared<op::v0::MatMul>(query, key_concat, false, true);
        qk->set_friendly_name("matmul1." + idx);
        auto add = std::make_shared<op::v1::Add>(qk->output(0), mask->output(0));
        add->set_friendly_name("add." + idx);
        auto softmax = std::make_shared<op::v8::Softmax>(add->output(0), 3);
        softmax->set_friendly_name("softmax." + idx);
        auto matmul2 = std::make_shared<op::v0::MatMul>(softmax->output(0), value_concat->output(0));
        matmul2->set_friendly_name("matmul2." + idx);

        // Transpose + Reshape tail required by SDPADecomposed pattern root (Reshape3)
        auto order = std::make_shared<op::v0::Constant>(element::i32, Shape{4}, std::vector<int32_t>{0, 2, 1, 3});
        auto transpose = std::make_shared<op::v1::Transpose>(matmul2->output(0), order->output(0));
        transpose->set_friendly_name("transpose." + idx);
        const int64_t out_size = static_cast<int64_t>(num_heads * head_dim);
        auto shape_const =
            std::make_shared<op::v0::Constant>(element::i64,
                                               Shape{3},
                                               std::vector<int64_t>{1, static_cast<int64_t>(query_len), out_size});
        auto reshape = std::make_shared<op::v1::Reshape>(transpose->output(0), shape_const->output(0), false);
        reshape->set_friendly_name("reshape." + idx);

        auto result = std::make_shared<op::v0::Result>(reshape->output(0));
        result->set_friendly_name("out." + idx);
        results.push_back(result);
    }

    auto model = std::make_shared<Model>(results, params, "decomposed_sdpa_model");
    model->validate_nodes_and_infer_types();
    return model;
}

// Build a trivial model (no SDPA structure) to verify the pattern does NOT fire
std::shared_ptr<ov::Model> build_non_attention_model() {
    using namespace ov;
    auto in = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 16});
    in->set_friendly_name("input");
    auto bias = std::make_shared<op::v0::Constant>(element::f32, Shape{1, 16}, std::vector<float>(16, 0.1f));
    auto add = std::make_shared<op::v1::Add>(in->output(0), bias->output(0));
    add->set_friendly_name("add");
    auto out = std::make_shared<op::v0::Result>(add->output(0));
    out->set_friendly_name("out");
    auto model = std::make_shared<Model>(ResultVector{out}, ParameterVector{in}, "non_attention_model");
    model->validate_nodes_and_infer_types();
    return model;
}

}  // namespace

// ---- SDPADecomposed pattern matching ----

// Verify that SDPADecomposed matches a single-layer standard decomposed SDPA model.
// Uses contiguous 2-input Concat (baseline behaviour, must remain working).
TEST(SdpaPatternMatcherTest, SDPADecomposedMatchesSingleLayerStandardModel) {
    auto model = build_decomposed_sdpa_model(/*num_layers=*/1);
    auto cfg = make_cfg({{"NPUW_ONLINE_PIPELINE", "REP"}, {"NPUW_ONLINE_ISOLATE", "ATTN"}});

    auto ens = ov::npuw::online::buildPartitioning(model, cfg);

    EXPECT_GE(count_groups_with_tag(ens, "attn"), 1u);
}

// Verify that SDPADecomposed matches every layer in a multi-layer model.
TEST(SdpaPatternMatcherTest, SDPADecomposedMatchesMultiLayerStandardModel) {
    constexpr size_t num_layers = 4;
    auto model = build_decomposed_sdpa_model(num_layers);
    auto cfg = make_cfg({{"NPUW_ONLINE_PIPELINE", "REP"}, {"NPUW_ONLINE_ISOLATE", "ATTN"}});

    auto ens = ov::npuw::online::buildPartitioning(model, cfg);

    // At minimum, one group per SDPA layer must carry the "attn" tag
    EXPECT_GE(count_groups_with_tag(ens, "attn"), num_layers);
}

// Verify that SDPADecomposed matches a model where the KV Concat has multiple block
// inputs (post-SplitKVCacheIntoBlocks layout, no Convert wrappers).
// This is the core scenario introduced by the unconstrained Concat change in our PR.
TEST(SdpaPatternMatcherTest, SDPADecomposedMatchesBlockKvCacheModel) {
    // 3 past key/value block params per layer -> Concat has 4 inputs (3 blocks + new token)
    auto model = build_decomposed_sdpa_model(/*num_layers=*/1, /*num_key_blocks=*/3);
    auto cfg = make_cfg({{"NPUW_ONLINE_PIPELINE", "REP"}, {"NPUW_ONLINE_ISOLATE", "ATTN"}});

    auto ens = ov::npuw::online::buildPartitioning(model, cfg);

    EXPECT_GE(count_groups_with_tag(ens, "attn"), 1u);
}

// Verify that SDPADecomposed does NOT tag groups in a model with no SDPA pattern.
TEST(SdpaPatternMatcherTest, SDPADecomposedDoesNotMatchNonAttentionModel) {
    auto model = build_non_attention_model();
    auto cfg = make_cfg({{"NPUW_ONLINE_PIPELINE", "REP"}, {"NPUW_ONLINE_ISOLATE", "ATTN"}});

    auto ens = ov::npuw::online::buildPartitioning(model, cfg);

    EXPECT_EQ(count_groups_with_tag(ens, "attn"), 0u);
}
