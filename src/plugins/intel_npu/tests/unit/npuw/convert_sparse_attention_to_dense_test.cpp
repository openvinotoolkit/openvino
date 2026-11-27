// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/convert_sparse_attention_to_dense.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/qsa_prepare.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/sparse_attention.hpp"
#include "openvino/op/sparse_attention_indexer.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/openvino.hpp"

namespace {

// S == max_kv_len, so the number of key blocks is static.
constexpr size_t kBlock = 4;         // key_block_size
constexpr size_t kSeq = 16;          // static KV length == max_kv_len
constexpr size_t kBlocks = 4;        // kSeq / kBlock
constexpr size_t kPastCapacity = 5;  // > kBlocks, to exercise the state resize
constexpr size_t kHeadDim = 8;       // Di
constexpr size_t kRotDim = 4;        // Dr
constexpr size_t kIndexHeads = 2;    // Hi
constexpr size_t kQueryHeads = 2;    // H
constexpr size_t kKVHeads = 1;       // Hkv
constexpr size_t kAttnDim = 8;       // E
constexpr size_t kValDim = 6;        // Ev

// StatefulToStateless keeps the QSA variables unrecognized under
// "{input|output}_restored." + <past_variable_id> + <present_variable_id>.
std::string past_state_name(const std::string& kind) {
    return "input_restored.cache_params.past." + kind + ".0cache_params.present." + kind + ".0";
}

std::string present_state_name(const std::string& kind) {
    return "output_restored.cache_params.past." + kind + ".0cache_params.present." + kind + ".0";
}

// The chunk length and the disabled features shape the decomposed graph.
struct Scenario {
    int64_t seq_len = 16;  // L of the current chunk
    bool rope = true;
    int64_t token_budget = 16;  // 16 selects all 4 blocks, 8 only 2
};

std::shared_ptr<ov::op::v0::Parameter> find_param(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    for (const auto& p : model->get_parameters()) {
        if (p->get_output_tensor(0).get_names().count(name) > 0) {
            return p;
        }
    }
    return nullptr;
}

std::shared_ptr<ov::op::v0::Result> find_result(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    for (const auto& r : model->get_results()) {
        if (r->get_output_tensor(0).get_names().count(name) > 0) {
            return r;
        }
    }
    return nullptr;
}

std::shared_ptr<ov::Model> build_model(const Scenario& sc) {
    ov::ParameterVector params;
    const auto param = [&](const ov::PartialShape& shape, const std::string& name) {
        auto p = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
        p->get_output_tensor(0).set_names({name});
        params.push_back(p);
        return p;
    };

    auto mask = param({1, kSeq}, "attention_mask");
    auto raw = param({1, sc.seq_len, kHeadDim}, "qsa/raw");
    auto past_blocks = param({1, kPastCapacity, kHeadDim}, past_state_name("block_index_key"));
    auto past_inc_raw = param({1, kBlock - 1, kHeadDim}, past_state_name("incomplete_index_key"));

    ov::Output<ov::Node> rope_cos, rope_sin, past_rope;
    if (sc.rope) {
        rope_cos = param({1, sc.seq_len, kRotDim}, "qsa/rope_cos")->output(0);
        rope_sin = param({1, sc.seq_len, kRotDim}, "qsa/rope_sin")->output(0);
        past_rope = param({1, 2 * kRotDim}, past_state_name("incomplete_index_rope"))->output(0);
    } else {
        // The rotation is disabled by empty constants, and the rope state does not exist
        const auto empty = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0});
        rope_cos = empty->output(0);
        rope_sin = empty->output(0);
        past_rope = empty->output(0);
    }
    auto k_norm = param({kHeadDim}, "qsa/k_norm");

    auto key_length =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{kSeq});
    auto prepare = std::make_shared<ov::op::v17::QSAPrepare>(raw->output(0),
                                                             past_blocks->output(0),
                                                             key_length->output(0),
                                                             rope_cos,
                                                             rope_sin,
                                                             k_norm->output(0),
                                                             past_inc_raw->output(0),
                                                             past_rope,
                                                             static_cast<int64_t>(kBlock),
                                                             1e-6f);

    auto index_query = param({1, kIndexHeads, sc.seq_len, kHeadDim}, "qsa/index_query");
    auto indexer = std::make_shared<ov::op::v17::SparseAttentionIndexer>(index_query->output(0),
                                                                         prepare->output(0),
                                                                         key_length->output(0),
                                                                         static_cast<int64_t>(kBlock),
                                                                         sc.token_budget,
                                                                         1);

    auto query = param({1, kQueryHeads, sc.seq_len, kAttnDim}, "qsa/query");
    auto key = param({1, kKVHeads, kSeq, kAttnDim}, "qsa/key");
    auto value = param({1, kKVHeads, kSeq, kValDim}, "qsa/value");
    auto attention = std::make_shared<ov::op::v17::SparseAttention>(query->output(0),
                                                                    key->output(0),
                                                                    value->output(0),
                                                                    indexer->output(0),
                                                                    indexer->output(1),
                                                                    static_cast<int64_t>(kBlock),
                                                                    1,
                                                                    true);
    attention->set_friendly_name("qsa/attention");

    const auto result_of = [&](const ov::Output<ov::Node>& out, const std::string& name) {
        auto res = std::make_shared<ov::op::v0::Result>(out);
        res->get_output_tensor(0).set_names({name});
        return res;
    };
    ov::ResultVector results{result_of(attention->output(0), "qsa/attention_out"),
                             result_of(indexer->output(0), "qsa/block_indices"),
                             result_of(indexer->output(1), "qsa/block_count"),
                             result_of(prepare->output(0), present_state_name("block_index_key")),
                             result_of(prepare->output(1), present_state_name("incomplete_index_key"))};
    if (sc.rope) {
        results.push_back(result_of(prepare->output(2), present_state_name("incomplete_index_rope")));
    }
    return std::make_shared<ov::Model>(results, params);
}

template <class T>
size_t count_nodes(const std::shared_ptr<ov::Model>& model) {
    size_t count = 0;
    for (const auto& node : model->get_ordered_ops()) {
        count += ov::is_type<T>(node) ? 1 : 0;
    }
    return count;
}

}  // namespace

TEST(NPUWConvertSparseAttentionToDense, DecomposesQSAPrepareStatesToLinearCache) {
    for (const bool rope : {true, false}) {
        SCOPED_TRACE(rope ? "rope" : "no rope");
        auto model = build_model(Scenario{16, rope, 16});
        EXPECT_TRUE(ov::npuw::ConvertSparseAttentionToDense(kSeq).run_on_model(model));
        EXPECT_EQ(count_nodes<ov::op::v17::QSAPrepare>(model), 0u);

        for (const auto& kind : {"block_index_key", "incomplete_index_key", "incomplete_index_rope"}) {
            if (!rope && std::string(kind) == "incomplete_index_rope") {
                // the rope state does not exist when the rotation is disabled
                EXPECT_EQ(find_param(model, "cache_params.past.incomplete_index_rope.0"), nullptr);
                EXPECT_EQ(find_result(model, "cache_params.present.incomplete_index_rope.0"), nullptr);
                continue;
            }
            EXPECT_NE(find_param(model, "cache_params.past." + std::string(kind) + ".0"), nullptr) << kind;
            EXPECT_NE(find_result(model, "cache_params.present." + std::string(kind) + ".0"), nullptr) << kind;
        }
        const auto blocks = find_param(model, "cache_params.past.block_index_key.0");
        ASSERT_NE(blocks, nullptr);
        EXPECT_EQ(blocks->get_partial_shape(),
                  (ov::PartialShape{1, static_cast<ov::Dimension>(kBlocks), static_cast<ov::Dimension>(kHeadDim)}));
        const auto present = find_result(model, "cache_params.present.block_index_key.0");
        ASSERT_NE(present, nullptr);
        EXPECT_EQ(present->get_input_partial_shape(0),
                  (ov::PartialShape{1, static_cast<ov::Dimension>(kBlocks), static_cast<ov::Dimension>(kHeadDim)}));
    }
}

TEST(NPUWConvertSparseAttentionToDense, DecomposesIndexerToTopKSelection) {
    const Scenario sc{4, true, 8};  // a chunked step, 2 of the 4 blocks selected
    auto model = build_model(sc);
    EXPECT_TRUE(ov::npuw::ConvertSparseAttentionToDense(kSeq).run_on_model(model));
    EXPECT_EQ(count_nodes<ov::op::v17::SparseAttentionIndexer>(model), 0u);
    EXPECT_GE(count_nodes<ov::op::v11::TopK>(model), 1u);

    const auto indices = find_result(model, "qsa/block_indices");
    ASSERT_NE(indices, nullptr);
    EXPECT_EQ(indices->get_input_element_type(0), ov::element::i32);
    EXPECT_EQ(indices->get_input_partial_shape(0),
              (ov::PartialShape{1,
                                static_cast<ov::Dimension>(1),
                                static_cast<ov::Dimension>(sc.seq_len),
                                static_cast<ov::Dimension>(sc.token_budget / kBlock)}));
    const auto count = find_result(model, "qsa/block_count");
    ASSERT_NE(count, nullptr);
    EXPECT_EQ(count->get_input_element_type(0), ov::element::i32);
    EXPECT_EQ(count->get_input_partial_shape(0),
              (ov::PartialShape{1, static_cast<ov::Dimension>(1), static_cast<ov::Dimension>(sc.seq_len)}));
}

TEST(NPUWConvertSparseAttentionToDense, ReplacesSparseAttentionWithDenseSDPA) {
    auto model = build_model(Scenario{});
    EXPECT_TRUE(ov::npuw::ConvertSparseAttentionToDense(kSeq).run_on_model(model));
    EXPECT_FALSE(ov::npuw::ConvertSparseAttentionToDense(kSeq).run_on_model(model));
    EXPECT_EQ(count_nodes<ov::op::v17::QSAPrepare>(model), 0u);
    EXPECT_EQ(count_nodes<ov::op::v17::SparseAttentionIndexer>(model), 0u);
    EXPECT_EQ(count_nodes<ov::op::v17::SparseAttention>(model), 0u);

    ASSERT_EQ(count_nodes<ov::op::v13::ScaledDotProductAttention>(model), 1u);
    for (const auto& node : model->get_ordered_ops()) {
        if (auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(node)) {
            EXPECT_EQ(sdpa->get_friendly_name(), "qsa/attention/npuw_dense_sdpa");
            EXPECT_EQ(sdpa->get_input_size(), 5u);  // query, key, value, mask, scale
        }
    }
}
