// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Frontend operations for block-sparse attention and hashed n-gram embeddings.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "cnpy.h"
#include "gtest/gtest.h"
#include "op_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/ngram_hash_function.hpp"
#include "openvino/op/qsa_indexer.hpp"
#include "openvino/op/sparse_attention.hpp"
#include "openvino/runtime/core.hpp"
#include "ov_ops/rms.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"

using namespace ov_gguf_test;

namespace {

template <class T>
std::shared_ptr<T> find_op(const std::shared_ptr<ov::Model>& model) {
    for (const auto& node : model->get_ordered_ops()) {
        if (auto typed = ov::as_type_ptr<T>(node))
            return typed;
    }
    return nullptr;
}

std::vector<ov::Tensor> run_all(const std::shared_ptr<ov::Model>& model,
                                const std::map<std::string, ov::Tensor>& inputs) {
    ov::Core core;
    auto request =
        core.compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32)).create_infer_request();
    for (const auto& [name, tensor] : inputs)
        request.set_tensor(name, tensor);
    request.infer();
    std::vector<ov::Tensor> outputs;
    for (const auto& output : model->outputs())
        outputs.push_back(request.get_tensor(output));
    return outputs;
}

}  // namespace

// llama.cpp qwen4exp: the token and its predecessors (EOS before the start and across an EOS)
// hashed with wrapping uint64 products, XOR and a per-head modulus.
TEST(GGUFSparseAttentionOps, NgramHashMatchesUint64Reference) {
    const int64_t ngram = 3, per_ngram = 2, eos = 248044;
    const std::vector<int64_t> multipliers{23703573157769, 20109073645365, 8052911324071};
    const std::vector<int64_t> vocab{1009, 16777213, 999983, 7};
    const std::vector<int64_t> offsets{0, 1009, 16778222, 17778205};
    const size_t batch = 2, tokens = 6, heads = 4, context = 2;
    const std::vector<int32_t> ids{248319, 5, eos, 17, 248000, 3, 1, 2, 3, eos, eos, 9};
    // Histories are stored relative to EOS: zero is a sequence start.
    const std::vector<int32_t> history{0, 0, 12 - eos, 248001 - eos};

    auto model = SingleOpBuilder()
                     .op("GGUF_OP_NGRAM_HASH")
                     .input("tokens", ov::element::i32, {1, 1, -1, -1})
                     .input("history", ov::element::i32, {-1, static_cast<int64_t>(context)})
                     .output("rows", ov::element::i32, {1, 1, -1, -1})
                     .extra_output("new_history")
                     .attr("ngram_size", ngram)
                     .attr("heads_per_ngram", per_ngram)
                     .attr("multipliers", multipliers)
                     .attr("head_vocab_sizes", vocab)
                     .attr("head_offsets", offsets)
                     .attr("eos_token_id", eos)
                     .attr("history_offset", eos)
                     .attr("token_limit", int64_t{248320})
                     .build();
    const auto hash = find_op<ov::op::internal::NGramHashFunction>(model);
    ASSERT_TRUE(hash);
    EXPECT_EQ(hash->get_heads_per_order(), (std::vector<int64_t>{2, 2}));
    EXPECT_EQ(hash->get_multipliers(), multipliers);
    ov::Tensor ids_tensor(ov::element::i32, {1, 1, batch, tokens});
    std::copy(ids.begin(), ids.end(), ids_tensor.data<int32_t>());
    ov::Tensor history_tensor(ov::element::i32, {batch, context});
    std::copy(history.begin(), history.end(), history_tensor.data<int32_t>());
    const auto outputs = run_all(model, {{"tokens", ids_tensor}, {"history", history_tensor}});
    ASSERT_EQ(outputs[0].get_shape(), (ov::Shape{1, 1, batch, tokens * heads}));
    ASSERT_EQ(outputs[1].get_shape(), (ov::Shape{batch, context}));

    for (size_t b = 0; b < batch; ++b) {
        std::vector<int64_t> sequence{history[b * context] + eos, history[b * context + 1] + eos};
        for (size_t t = 0; t < tokens; ++t)
            sequence.push_back(ids[b * tokens + t]);
        for (size_t t = 0; t < tokens; ++t) {
            std::vector<int64_t> window(ngram);
            window[0] = sequence[context + t];
            bool cut = false;
            for (int64_t s = 1; s < ngram; ++s) {
                const auto token = sequence[context + t - s];
                window[s] = cut ? eos : token;
                cut = cut || token == eos;
            }
            for (int64_t order = 2; order <= ngram; ++order) {
                uint64_t mixed = static_cast<uint64_t>(window[0]) * static_cast<uint64_t>(multipliers[0]);
                for (int64_t j = 1; j < order; ++j)
                    mixed ^= static_cast<uint64_t>(window[j]) * static_cast<uint64_t>(multipliers[j]);
                for (int64_t e = 0; e < per_ngram; ++e) {
                    const auto h = static_cast<size_t>((order - 2) * per_ngram + e);
                    const auto expected = static_cast<int32_t>(mixed % static_cast<uint64_t>(vocab[h]) +
                                                               static_cast<uint64_t>(offsets[h]));
                    EXPECT_EQ(outputs[0].data<int32_t>()[(b * tokens + t) * heads + h], expected)
                        << "batch " << b << " token " << t << " head " << h;
                }
            }
        }
        for (size_t c = 0; c < context; ++c)
            EXPECT_EQ(outputs[1].data<int32_t>()[b * context + c], sequence[tokens + c] - eos);
    }
}

// NGramHashFunction takes a signed modulo of the 64-bit mix, which equals the unsigned one only
// while every token product stays below 2^63.
TEST(GGUFSparseAttentionOps, NgramHashRejectsProductsThatReachTheSignBit) {
    for (const int64_t limit : {int64_t{1} << 20, int64_t{1} << 31}) {
        SCOPED_TRACE("token_limit=" + std::to_string(limit));
        auto builder = SingleOpBuilder()
                           .op("GGUF_OP_NGRAM_HASH")
                           .input("tokens", ov::element::i32, {1, 1, -1, -1})
                           .input("history", ov::element::i32, {-1, 1})
                           .output("rows", ov::element::i32, {1, 1, -1, -1})
                           .extra_output("new_history")
                           .attr("ngram_size", int64_t{2})
                           .attr("heads_per_ngram", int64_t{1})
                           .attr("multipliers", std::vector<int64_t>{int64_t{1} << 42, 3})
                           .attr("head_vocab_sizes", std::vector<int64_t>{1009})
                           .attr("head_offsets", std::vector<int64_t>{0})
                           .attr("eos_token_id", int64_t{2})
                           .attr("token_limit", limit);
        if (limit < (int64_t{1} << 21))
            EXPECT_NO_THROW(builder.build());
        else
            EXPECT_THROW(builder.build(), ov::Exception);
    }
}

TEST(GGUFSparseAttentionOps, KeyLengthCountsKeysVisibleToTheLastQuery) {
    auto model = SingleOpBuilder()
                     .op("GGUF_OP_ATTENTION_KEY_LENGTH")
                     .input("mask", ov::element::f32, {-1, 1, -1, -1})
                     .output("length", ov::element::i32, {-1})
                     .build();
    const float x = -std::numeric_limits<float>::infinity();
    // Two queries over five keys; the second sequence has two padding keys on the left.
    ov::Tensor mask = make_f32_tensor({2, 1, 2, 5}, {0, 0, 0, 0, x, 0, 0, 0, 0, 0, x, x, 0, 0, x, x, x, 0, 0, 0});
    const auto outputs = run_all(model, {{"mask", mask}});
    ASSERT_EQ(outputs[0].get_shape(), (ov::Shape{2}));
    EXPECT_EQ(outputs[0].data<int32_t>()[0], 5);
    EXPECT_EQ(outputs[0].data<int32_t>()[1], 3);
}

// Interleaved M-RoPE tables in the rotate-half layout: frequency j uses the H axis for j % 3 == 1,
// the W axis for j % 3 == 2 (within their sections) and the T axis otherwise.
TEST(GGUFSparseAttentionOps, RopeCacheProducesRotateHalfImropeTables) {
    ov::frontend::gguf::RopeConfig rope;
    rope.n_dims = 12;
    rope.freq_base = 10000.0f;
    rope.freq_scale = 1.0f;
    rope.attn_factor = 1.0f;
    rope.is_imrope = true;
    rope.sections = {2, 2, 2, 0};
    auto model = SingleOpBuilder()
                     .op("GGUF_OP_ROPE_CACHE")
                     .input("pos", ov::element::i32, {1, 1, 1, -1})
                     .output("cos", ov::element::f32, {1, -1, 12})
                     .extra_output("sin")
                     .attr("rope_config", rope)
                     .build();
    const size_t tokens = 3;
    // Section-major positions: T, H, W, extra.
    const std::vector<int32_t> positions{0, 1, 2, 5, 6, 7, 9, 10, 11, 0, 0, 0};
    ov::Tensor pos(ov::element::i32, {1, 1, 1, 4 * tokens});
    std::copy(positions.begin(), positions.end(), pos.data<int32_t>());
    const auto outputs = run_all(model, {{"pos", pos}});
    ASSERT_EQ(outputs[0].get_shape(), (ov::Shape{1, tokens, 12}));
    for (size_t t = 0; t < tokens; ++t) {
        for (size_t j = 0; j < 6; ++j) {
            const size_t axis = j % 3 == 1 && j < 6 ? 1 : j % 3 == 2 && j < 6 ? 2 : 0;
            const double theta = positions[axis * tokens + t] * std::pow(10000.0, -2.0 * j / 12.0);
            for (size_t half : {0, 6}) {
                EXPECT_NEAR(outputs[0].data<float>()[t * 12 + half + j], std::cos(theta), 1e-5);
                EXPECT_NEAR(outputs[1].data<float>()[t * 12 + half + j], std::sin(theta), 1e-5);
            }
        }
    }
}

// QSAIndexer is emitted directly, with the index query norm and partial RoPE inside it, and grows
// the block-key cache to the complete blocks of the longest row.
TEST(GGUFSparseAttentionOps, QsaIndexerAppliesQueryNormAndRopeAndGrowsTheBlockCache) {
    using Indexer = ov::op::internal::QSAIndexer;
    const int64_t heads = 2, head_dim = 12, rotary = 8, block = 4;
    auto model = SingleOpBuilder()
                     .op("GGUF_OP_QSA_INDEXER")
                     .input("query", ov::element::f32, {1, 1, -1, heads * head_dim})
                     .input("q_norm", ov::element::f32, {1, 1, 1, head_dim})
                     .input("keys", ov::element::f32, {1, 1, -1, head_dim})
                     .input("k_norm", ov::element::f32, {1, 1, 1, head_dim})
                     .input("cos", ov::element::f32, {1, -1, rotary})
                     .input("sin", ov::element::f32, {1, -1, rotary})
                     .input("length", ov::element::i32, {1})
                     .input("blocks", ov::element::f32, {1, -1, head_dim})
                     .input("pending", ov::element::f32, {1, block - 1, head_dim})
                     .input("pending_rope", ov::element::f32, {1, 2 * rotary})
                     .output("indices", ov::element::i32, {1, 1, -1, 2})
                     .extra_output("count")
                     .extra_output("blocks_out")
                     .extra_output("pending_out")
                     .extra_output("pending_rope_out")
                     .attr("heads", heads)
                     .attr("key_block_size", block)
                     .attr("token_budget", int64_t{8})
                     .attr("eps", 1e-6f)
                     .build();
    const auto indexer = find_op<Indexer>(model);
    ASSERT_TRUE(indexer);
    const auto& config = indexer->get_config();
    EXPECT_TRUE(config.query_norm);
    EXPECT_TRUE(config.query_rope);
    EXPECT_EQ(config.key_block_size, 4u);
    EXPECT_EQ(config.block_topk, 2u);
    EXPECT_EQ(config.num_selection_heads, 1u);
    EXPECT_EQ(config.indexer_rotary_dim, 8u);
    EXPECT_EQ(config.indexer_head_dim, 12u);
    EXPECT_FLOAT_EQ(config.scale, 1.0f / std::sqrt(12.0f));
    EXPECT_EQ(indexer->get_input_partial_shape(Indexer::InputIdx::INDEX_QUERY), (ov::PartialShape{1, -1, 2, 12}));
    EXPECT_EQ(indexer->get_input_partial_shape(Indexer::InputIdx::RAW_INDEX_KEYS), (ov::PartialShape{1, -1, 1, 12}));
    EXPECT_EQ(indexer->get_input_partial_shape(Indexer::InputIdx::Q_NORM_WEIGHT), (ov::PartialShape{12}));
    EXPECT_EQ(model->get_output_size(), 5u);
    EXPECT_EQ(model->output(3).get_partial_shape(), (ov::PartialShape{1, 3, 12}));
    for (const auto& node : model->get_ops())
        EXPECT_FALSE(ov::is_type<ov::op::internal::RMS>(node) || ov::is_type<ov::op::internal::RoPE>(node));

    const auto filled = [](const ov::Shape& shape, float scale) {
        ov::Tensor t(ov::element::f32, shape);
        for (size_t i = 0; i < t.get_size(); ++i)
            t.data<float>()[i] = std::sin(scale * static_cast<float>(i + 1));
        return t;
    };
    const auto length = [](int32_t value) {
        ov::Tensor t(ov::element::i32, {1});
        t.data<int32_t>()[0] = value;
        return t;
    };
    // Prefill of 6 tokens into empty states, then 4 more tokens.
    std::map<std::string, ov::Tensor> inputs{{"q_norm", filled({1, 1, 1, 12}, 0.3f)},
                                             {"k_norm", filled({1, 1, 1, 12}, 0.7f)},
                                             {"blocks", ov::Tensor(ov::element::f32, {1, 0, 12})},
                                             {"pending", ov::Tensor(ov::element::f32, {1, 3, 12})},
                                             {"pending_rope", ov::Tensor(ov::element::f32, {1, 16})}};
    std::memset(inputs["pending"].data(), 0, inputs["pending"].get_byte_size());
    std::memset(inputs["pending_rope"].data(), 0, inputs["pending_rope"].get_byte_size());
    for (const auto& [tokens, total] : {std::pair<size_t, int32_t>{6, 6}, std::pair<size_t, int32_t>{4, 10}}) {
        SCOPED_TRACE("key_length=" + std::to_string(total));
        inputs["query"] = filled({1, 1, tokens, 24}, 1.1f + total);
        inputs["keys"] = filled({1, 1, tokens, 12}, 0.9f + total);
        inputs["cos"] = filled({1, tokens, 8}, 0.2f + total);
        inputs["sin"] = filled({1, tokens, 8}, 0.4f + total);
        inputs["length"] = length(total);
        const auto outputs = run_all(model, inputs);
        ASSERT_EQ(outputs[0].get_shape(), (ov::Shape{1, 1, tokens, 2}));
        ASSERT_EQ(outputs[1].get_shape(), (ov::Shape{1, 1, tokens}));
        EXPECT_EQ(outputs[2].get_shape(), (ov::Shape{1, static_cast<size_t>(total / block), 12}));
        EXPECT_EQ(outputs[3].get_shape(), (ov::Shape{1, 3, 12}));
        EXPECT_EQ(outputs[4].get_shape(), (ov::Shape{1, 16}));
        for (size_t t = 0; t < tokens; ++t) {
            const auto count = outputs[1].data<int32_t>()[t];
            EXPECT_GE(count, 0);
            EXPECT_LE(count, 2);
        }
        for (size_t i = 0; i < outputs[2].get_size(); ++i)
            ASSERT_TRUE(std::isfinite(outputs[2].data<float>()[i]));
        for (const auto& [name, output] :
             {std::pair{"blocks", 2}, std::pair{"pending", 3}, std::pair{"pending_rope", 4}}) {
            inputs[name] = ov::Tensor(outputs[output].get_element_type(), outputs[output].get_shape());
            outputs[output].copy_to(inputs[name]);
        }
    }
}

TEST(GGUFSparseAttentionOps, SparseAttentionUsesTheDefaultScaleUnlessItDiffers) {
    for (const float scale : {0.25f, 0.5f}) {
        auto model = SingleOpBuilder()
                         .op("GGUF_OP_SPARSE_ATTENTION")
                         .input("q", ov::element::f32, {1, -1, 4, 16})
                         .input("k", ov::element::f32, {1, 2, -1, 16})
                         .input("v", ov::element::f32, {1, 2, -1, 16})
                         .input("indices", ov::element::i32, {1, 1, -1, 2})
                         .input("count", ov::element::i32, {1, 1, -1})
                         .input("length", ov::element::i32, {1})
                         .output("out", ov::element::f32, {1, -1, 4, 16})
                         .attr("key_block_size", int64_t{4})
                         .attr("scale", scale)
                         .build();
        const auto attention = find_op<ov::op::v17::SparseAttention>(model);
        ASSERT_TRUE(attention);
        EXPECT_EQ(attention->get_key_block_size(), 4);
        EXPECT_TRUE(attention->get_causal());
        EXPECT_EQ(attention->get_input_size(), scale == 0.25f ? 6u : 7u);
        EXPECT_EQ(model->output(0).get_partial_shape(), (ov::PartialShape{1, -1, 4, 16}));
    }
}

namespace {

struct GgmlReference {
    cnpy::npz_t arrays = cnpy::npz_load(test_data_dir() + "/sparse_ops.npz");

    const cnpy::NpyArray& array(const std::string& name) const {
        return npz_array(arrays, name);
    }
    ov::element::Type type(const std::string& name) const {
        const auto& a = array(name);
        return a.word_size == 2 ? ov::element::f16
               : a.word_size == 4 && name.find("top_k") == std::string::npos &&
                       name.find("blocks") == std::string::npos && name.find("extra") == std::string::npos
                   ? ov::element::f32
                   : ov::element::i32;
    }
    ov::Shape shape(const std::string& name) const {
        return ov::Shape(array(name).shape.begin(), array(name).shape.end());
    }
    ov::Tensor tensor(const std::string& name) const {
        ov::Tensor t(type(name), shape(name));
        std::memcpy(t.data(), array(name).data<char>(), t.get_byte_size());
        return t;
    }
    std::vector<float> values(const std::string& name) const {
        const auto* data = array(name).data<float>();
        return std::vector<float>(data, data + ov::shape_size(shape(name)));
    }
};

void expect_same_with_infinities(const ov::Tensor& actual, const std::vector<float>& expected, float atol) {
    ASSERT_EQ(actual.get_size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        const float value = actual.data<float>()[i];
        if (std::isinf(expected[i]))
            EXPECT_EQ(value, expected[i]) << "at " << i;
        else
            EXPECT_NEAR(value, expected[i], atol + 1e-5f * std::abs(expected[i])) << "at " << i;
    }
}

}  // namespace

TEST(GGUFSparseAttentionOps, LightningIndexerMatchesGgmlCpu) {
    GgmlReference ref;
    auto model = SingleOpBuilder()
                     .op("GGML_OP_LIGHTNING_INDEXER")
                     .input("q", ov::element::f32, ref.shape("indexer.q"))
                     .input("k", ov::element::f32, ref.shape("indexer.k"))
                     .input("w", ov::element::f32, ref.shape("indexer.w"))
                     .input("mask", ov::element::f16, ref.shape("indexer.mask"))
                     .output("out", ov::element::f32, ref.shape("indexer.out"))
                     .build();
    const auto out = run_on_cpu(model,
                                {{"q", ref.tensor("indexer.q")},
                                 {"k", ref.tensor("indexer.k")},
                                 {"w", ref.tensor("indexer.w")},
                                 {"mask", ref.tensor("indexer.mask")}});
    expect_same_with_infinities(out, ref.values("indexer.out"), 1e-4f);
}

TEST(GGUFSparseAttentionOps, TopKMaskMatchesGgmlCpu) {
    GgmlReference ref;
    for (const bool optional : {true, false}) {
        SCOPED_TRACE(optional ? "score, blocks and extra" : "picks are cells");
        SingleOpBuilder builder;
        builder.op("GGML_OP_TOP_K_MASK")
            .input("mask", ov::element::f32, ref.shape("topk.mask"))
            .input("top_k", ov::element::i32, ref.shape("topk.top_k"))
            .output("out", ov::element::f32, ref.shape("topk.out"));
        std::map<std::string, ov::Tensor> inputs{{"mask", ref.tensor("topk.mask")},
                                                 {"top_k", ref.tensor("topk.top_k")}};
        if (optional) {
            for (const auto* name : {"score", "blocks", "extra"}) {
                const auto key = std::string("topk.") + name;
                builder.input(name, ref.type(key), ref.shape(key));
                inputs[name] = ref.tensor(key);
            }
        } else {
            builder.attr("has_score", false).attr("has_blocks", false).attr("has_extra", false);
        }
        const auto out = run_on_cpu(builder.build(), inputs);
        expect_same_with_infinities(out, ref.values(optional ? "topk.out" : "topk.plain"), 0.0f);
    }
}

TEST(GGUFSparseAttentionOps, HyperConnectionReadMatchesGgmlCpu) {
    GgmlReference ref;
    for (const bool gated : {false, true}) {
        SCOPED_TRACE(gated ? "gated" : "weighted");
        const std::string weights = gated ? "hc_pre.gate" : "hc_pre.w";
        SingleOpBuilder builder;
        builder.op("GGML_OP_DSV4_HC_PRE")
            .input("x", ov::element::f32, ref.shape("hc_pre.x"))
            .input("w", ov::element::f32, ref.shape(weights))
            .output("out", ov::element::f32, ref.shape("hc_pre.out"));
        if (gated)
            builder.attr("scale", 0.25f).attr("gated", true);
        const auto out = run_on_cpu(builder.build(), {{"x", ref.tensor("hc_pre.x")}, {"w", ref.tensor(weights)}});
        expect_near(out, ref.values(gated ? "hc_pre.gated" : "hc_pre.out"));
    }
}

TEST(GGUFSparseAttentionOps, HyperConnectionWriteMatchesGgmlCpu) {
    GgmlReference ref;
    for (const bool mixed : {true, false}) {
        SCOPED_TRACE(mixed ? "stream mixing" : "identity");
        SingleOpBuilder builder;
        builder.op("GGML_OP_DSV4_HC_POST")
            .input("x", ov::element::f32, ref.shape("hc_post.x"))
            .input("residual", ov::element::f32, ref.shape("hc_post.residual"))
            .input("post", ov::element::f32, ref.shape("hc_post.post"))
            .output("out", ov::element::f32, ref.shape("hc_post.out"));
        std::map<std::string, ov::Tensor> inputs{{"x", ref.tensor("hc_post.x")},
                                                 {"residual", ref.tensor("hc_post.residual")},
                                                 {"post", ref.tensor("hc_post.post")}};
        if (mixed) {
            builder.input("comb", ov::element::f32, ref.shape("hc_post.comb"));
            inputs["comb"] = ref.tensor("hc_post.comb");
        }
        const auto out = run_on_cpu(builder.build(), inputs);
        expect_near(out, ref.values(mixed ? "hc_post.out" : "hc_post.identity"));
    }
}
