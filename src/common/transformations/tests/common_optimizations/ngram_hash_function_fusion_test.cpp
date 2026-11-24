// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/ngram_hash_function_fusion.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/bitwise_xor.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/ngram_hash_function.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"

using namespace ov;
using namespace ov::op;

namespace {

const std::vector<int64_t> kQwenMultipliers{23703573157769, 20109073645365, 8052911324071, 3000000019};
const std::vector<int64_t> kWrapMultipliers{0x3BF58476D1CE4E5B,
                                            static_cast<int64_t>(0x9E3779B97F4A7C15ULL),
                                            1234567891011,
                                            static_cast<int64_t>(0xD6E8FEB86659FD93ULL)};

struct HashGraphConfig {
    std::string name;
    std::vector<int64_t> multipliers;      // multipliers[k] for the token k steps back
    std::vector<int64_t> heads_per_order;  // orders 2 .. multipliers.size(), 0 = order not emitted
    PartialShape token_shape;
    bool share_mix = true;  // reuse the Xor of the previous order (HF) or recompute every order
};

struct HashTables {
    std::vector<int64_t> vocab_sizes;
    std::vector<int64_t> offsets;
};

HashTables make_tables(const std::vector<int64_t>& heads_per_order, int64_t base_vocab = 20000003) {
    HashTables t;
    int64_t offset = 0;
    for (const auto heads : heads_per_order) {
        for (int64_t e = 0; e < heads; ++e) {
            const int64_t vocab = base_vocab + static_cast<int64_t>(t.vocab_sizes.size()) * 20;
            t.vocab_sizes.push_back(vocab);
            t.offsets.push_back(offset);
            offset += vocab;
        }
    }
    return t;
}

ParameterVector make_tokens(const HashGraphConfig& c, element::Type type = element::i64) {
    ParameterVector tokens;
    for (size_t k = 0; k < c.multipliers.size(); ++k) {
        tokens.push_back(std::make_shared<v0::Parameter>(type, c.token_shape));
        tokens.back()->set_friendly_name("token_back_" + std::to_string(k));
    }
    return tokens;
}

// The traced Qwen4ExpTextNGramEmbedding / Engram hash: per order an XOR chain, FloorMod by the head vocab sizes,
// offsets, and a Concat of all orders on the last axis.
std::shared_ptr<Model> make_hash_model(const HashGraphConfig& c,
                                       const HashTables& t,
                                       element::Type type = element::i64,
                                       int64_t unsqueeze_axis = -1,
                                       bool reverse_concat = false) {
    const auto tokens = make_tokens(c, type);
    auto term = [&](size_t k) {
        return std::make_shared<v1::Multiply>(tokens[k], v0::Constant::create(type, Shape{}, {c.multipliers[k]}));
    };
    OutputVector blocks;
    Output<Node> mix = term(0);
    size_t h = 0;
    for (size_t k = 1; k < c.multipliers.size(); ++k) {
        if (c.share_mix) {
            mix = std::make_shared<v13::BitwiseXor>(mix, term(k));
        } else {
            // Recompute from scratch, newest term first.
            mix = term(k);
            for (size_t j = 0; j < k; ++j) {
                mix = std::make_shared<v13::BitwiseXor>(term(j), mix);
            }
        }
        const auto heads = static_cast<size_t>(c.heads_per_order[k - 1]);
        if (heads == 0) {
            continue;
        }
        const std::vector<int64_t> vocab(t.vocab_sizes.begin() + h, t.vocab_sizes.begin() + h + heads);
        const std::vector<int64_t> offs(t.offsets.begin() + h, t.offsets.begin() + h + heads);
        h += heads;
        auto unsq = std::make_shared<v0::Unsqueeze>(mix, v0::Constant::create(element::i64, Shape{}, {unsqueeze_axis}));
        // Like head_vocab_sizes.view(1, 1, -1): broadcast along the last axis only.
        Shape vocab_shape(c.token_shape.size() + 1, 1);
        vocab_shape.back() = heads;
        auto mod = std::make_shared<v1::FloorMod>(unsq, v0::Constant::create(type, vocab_shape, vocab));
        blocks.push_back(std::make_shared<v1::Add>(mod, v0::Constant::create(type, Shape{heads}, offs)));
    }
    if (reverse_concat) {
        std::reverse(blocks.begin(), blocks.end());
    }
    Output<Node> out = blocks.size() == 1 ? blocks[0] : std::make_shared<v0::Concat>(blocks, -1)->output(0);
    return std::make_shared<Model>(OutputVector{out}, tokens);
}

// The expected rewrite: Convert(NGramHashFunction(Concat(Unsqueeze(x[n-1]), ..., Unsqueeze(x[0])))).
Output<Node> make_fused_hash(const OutputVector& tokens,
                             const std::vector<int64_t>& multipliers,
                             const std::vector<int64_t>& heads_per_order,
                             const std::vector<int64_t>& vocab_sizes,
                             const std::vector<int64_t>& offsets) {
    const auto axis = v0::Constant::create(element::i64, Shape{1}, {-1});
    OutputVector window;
    for (auto it = tokens.rbegin(); it != tokens.rend(); ++it) {
        window.push_back(std::make_shared<v0::Unsqueeze>(*it, axis));
    }
    const auto hash = std::make_shared<ov::op::internal::NGramHashFunction>(std::make_shared<v0::Concat>(window, -1),
                                                                            heads_per_order,
                                                                            multipliers,
                                                                            vocab_sizes,
                                                                            offsets);
    return std::make_shared<v0::Convert>(hash, element::i64);
}

std::shared_ptr<Model> make_fused_model(const HashGraphConfig& c, const HashTables& t) {
    const auto tokens = make_tokens(c);
    const auto out = make_fused_hash(OutputVector(tokens.begin(), tokens.end()),
                                     c.multipliers,
                                     c.heads_per_order,
                                     t.vocab_sizes,
                                     t.offsets);
    return std::make_shared<Model>(OutputVector{out}, tokens);
}

// A single-order hash as produced by NGramHashBlockFusion: heads only for the last order.
Output<Node> make_fused_order(const OutputVector& tokens,
                              const std::vector<int64_t>& multipliers,
                              const std::vector<int64_t>& vocab_sizes,
                              const std::vector<int64_t>& offsets) {
    std::vector<int64_t> heads_per_order(tokens.size() - 1, 0);
    heads_per_order.back() = static_cast<int64_t>(vocab_sizes.size());
    return make_fused_hash(tokens, multipliers, heads_per_order, vocab_sizes, offsets);
}

std::vector<int64_t> head_range(const std::vector<int64_t>& values, size_t begin, size_t end) {
    return {values.begin() + begin, values.begin() + end};
}

// Compares both models on inputs that wrap the 64-bit products and produce negative mixes.
void expect_same_outputs(const std::shared_ptr<Model>& actual, const std::shared_ptr<Model>& expected) {
    ov::TensorVector inputs;
    for (const auto& parameter : actual->get_parameters()) {
        Shape shape;
        for (const auto& dimension : parameter->get_partial_shape()) {
            shape.push_back(dimension.is_static() ? dimension.get_length() : 2);
        }
        Tensor tensor(element::i64, shape);
        const auto values = tensor.data<int64_t>();
        for (size_t i = 0; i < tensor.get_size(); ++i) {
            values[i] = i % 2 ? std::numeric_limits<int64_t>::min() : 3000000000LL + static_cast<int64_t>(i);
        }
        inputs.push_back(tensor);
    }
    const auto lhs = ov::test::utils::infer_on_template(actual->clone(), inputs);
    const auto rhs = ov::test::utils::infer_on_template(expected->clone(), inputs);
    ASSERT_EQ(lhs.size(), rhs.size());
    for (size_t i = 0; i < lhs.size(); ++i) {
        ASSERT_EQ(lhs[i].get_shape(), rhs[i].get_shape());
        ASSERT_EQ(lhs[i].get_element_type(), element::i64);
        EXPECT_EQ(std::vector<int64_t>(lhs[i].data<int64_t>(), lhs[i].data<int64_t>() + lhs[i].get_size()),
                  std::vector<int64_t>(rhs[i].data<int64_t>(), rhs[i].data<int64_t>() + rhs[i].get_size()));
    }
}

// Replaces every Unsqueeze with Reshape(x, pattern).
void replace_unsqueeze_with_reshape(const std::shared_ptr<Model>& model,
                                    const std::vector<int64_t>& pattern,
                                    bool special_zero) {
    for (const auto& node : model->get_ordered_ops()) {
        if (const auto unsqueeze = ov::as_type_ptr<v0::Unsqueeze>(node)) {
            ov::replace_node(
                unsqueeze,
                std::make_shared<v1::Reshape>(unsqueeze->input_value(0),
                                              v0::Constant::create(element::i64, Shape{pattern.size()}, pattern),
                                              special_zero));
        }
    }
    model->validate_nodes_and_infer_types();
}

const std::vector<HashGraphConfig> kPositiveCases{
    {"Qwen3_8FlashNext_Bigram", {kQwenMultipliers[0], kQwenMultipliers[1]}, {8}, {-1, -1}},
    {"Qwen3_8FlashNext_Trigram", {kQwenMultipliers.begin(), kQwenMultipliers.begin() + 3}, {8, 8}, {-1, -1}},
    {"Qwen3_8FlashNext_Trigram_ContinuousBatching",
     {kQwenMultipliers.begin(), kQwenMultipliers.begin() + 3},
     {8, 8},
     {-1}},
    {"DeepSeekV4_1Flash_Bigram", {kWrapMultipliers[0], kWrapMultipliers[1]}, {4}, {-1}},
    {"DeepSeekV4_1Flash_Trigram", {kWrapMultipliers.begin(), kWrapMultipliers.begin() + 3}, {4, 4}, {-1}},
    {"DeepSeekV4_1Flash_Trigram_RecomputedMix",
     {kWrapMultipliers.begin(), kWrapMultipliers.begin() + 3},
     {4, 4},
     {-1},
     false},
    {"Fourgram_PerOrderHeads", kWrapMultipliers, {2, 3, 1}, {-1, 7}},
    {"Fourgram_SkippedTrigram", kQwenMultipliers, {2, 0, 3}, {-1}},
};

const HashGraphConfig kBigram{"Bigram", {11, 13}, {2}, {-1, -1}};
const HashGraphConfig kTrigram{"Trigram", {kWrapMultipliers.begin(), kWrapMultipliers.begin() + 3}, {4, 4}, {-1, -1}};

}  // namespace

class NGramHashFunctionFusionTests : public TransformationTestsF {
public:
    NGramHashFunctionFusionTests() {
        comparator.enable(FunctionsComparator::CONST_VALUES);
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<ov::pass::NGramHashFunctionFusion>();
    }
};

class NGramHashFunctionFusionParamTests : public NGramHashFunctionFusionTests,
                                          public testing::WithParamInterface<HashGraphConfig> {
public:
    static std::string get_test_case_name(const testing::TestParamInfo<HashGraphConfig>& info) {
        return info.param.name;
    }
};

TEST_P(NGramHashFunctionFusionParamTests, FusesAllOrdersIntoOneOp) {
    const auto& c = GetParam();
    const auto t = make_tables(c.heads_per_order);
    model = make_hash_model(c, t);
    model_ref = make_fused_model(c, t);
    expect_same_outputs(model, model_ref);
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         NGramHashFunctionFusionParamTests,
                         testing::ValuesIn(kPositiveCases),
                         NGramHashFunctionFusionParamTests::get_test_case_name);

TEST_F(NGramHashFunctionFusionTests, DoesNotFuseI32) {
    const HashGraphConfig i32_trigram{"I32Trigram", {1000003, 999983, 7}, {4, 4}, {-1, -1}};
    model = make_hash_model(i32_trigram, make_tables(i32_trigram.heads_per_order, 1000003), element::i32);
}

TEST_F(NGramHashFunctionFusionTests, DoesNotFuseConcatWhenRowsOverflowInt32) {
    // Only the bigram block, whose rows fit int32, becomes an op; the trigram block and the Concat stay.
    auto t = make_tables(kTrigram.heads_per_order);
    t.offsets.back() = std::numeric_limits<int32_t>::max() - t.vocab_sizes.back() + 2;
    model = make_hash_model(kTrigram, t);
    {
        const auto tokens = make_tokens(kTrigram);
        const auto& m = kTrigram.multipliers;
        auto term = [&](size_t k) {
            return std::make_shared<v1::Multiply>(tokens[k], v0::Constant::create(element::i64, Shape{}, {m[k]}));
        };
        const auto bigram = make_fused_order({tokens[0], tokens[1]},
                                             head_range(m, 0, 2),
                                             head_range(t.vocab_sizes, 0, 4),
                                             head_range(t.offsets, 0, 4));
        const auto mix =
            std::make_shared<v13::BitwiseXor>(std::make_shared<v13::BitwiseXor>(term(0), term(1)), term(2));
        const auto unsq = std::make_shared<v0::Unsqueeze>(mix, v0::Constant::create(element::i64, Shape{}, {-1}));
        const auto mod = std::make_shared<v1::FloorMod>(
            unsq,
            v0::Constant::create(element::i64, Shape{1, 1, 4}, head_range(t.vocab_sizes, 4, 8)));
        const auto trigram =
            std::make_shared<v1::Add>(mod, v0::Constant::create(element::i64, Shape{4}, head_range(t.offsets, 4, 8)));
        model_ref =
            std::make_shared<Model>(OutputVector{std::make_shared<v0::Concat>(OutputVector{bigram, trigram}, -1)},
                                    tokens);
    }
}

TEST_F(NGramHashFunctionFusionTests, DoesNotFuseUnsqueezeOnOtherAxis) {
    model = make_hash_model(kTrigram, make_tables(kTrigram.heads_per_order), element::i64, 0);
}

TEST_F(NGramHashFunctionFusionTests, FusesEveryOrderOnItsOwnWhenConcatOrderIsDescending) {
    // Heads must be grouped by ascending order to fit one op; otherwise every block is fused separately.
    const auto t = make_tables(kTrigram.heads_per_order);
    model = make_hash_model(kTrigram, t, element::i64, -1, true);
    {
        const auto tokens = make_tokens(kTrigram);
        const auto& m = kTrigram.multipliers;
        const auto trigram = make_fused_order({tokens[0], tokens[1], tokens[2]},
                                              m,
                                              head_range(t.vocab_sizes, 4, 8),
                                              head_range(t.offsets, 4, 8));
        const auto bigram = make_fused_order({tokens[0], tokens[1]},
                                             head_range(m, 0, 2),
                                             head_range(t.vocab_sizes, 0, 4),
                                             head_range(t.offsets, 0, 4));
        model_ref =
            std::make_shared<Model>(OutputVector{std::make_shared<v0::Concat>(OutputVector{trigram, bigram}, -1)},
                                    tokens);
    }
}

TEST_F(NGramHashFunctionFusionTests, DoesNotFuseConcatWhenHigherOrderDoesNotExtendLowerOrder) {
    // The trigram uses a different second token than the bigram: not one n-gram window, so the blocks are fused
    // on their own.
    const auto& m = kTrigram.multipliers;
    {
        auto tokens = make_tokens(kTrigram);
        auto extra = std::make_shared<v0::Parameter>(element::i64, kTrigram.token_shape);
        tokens.push_back(extra);
        auto mul = [](const Output<Node>& x, int64_t multiplier) {
            return std::make_shared<v1::Multiply>(x, v0::Constant::create(element::i64, Shape{}, {multiplier}));
        };
        auto block = [](const Output<Node>& mix, int64_t vocab, int64_t offset) {
            auto unsq = std::make_shared<v0::Unsqueeze>(mix, v0::Constant::create(element::i64, Shape{}, {-1}));
            auto mod = std::make_shared<v1::FloorMod>(unsq, v0::Constant::create(element::i64, Shape{1}, {vocab}));
            return std::make_shared<v1::Add>(mod, v0::Constant::create(element::i64, Shape{1}, {offset}));
        };
        auto bigram = std::make_shared<v13::BitwiseXor>(mul(tokens[0], m[0]), mul(tokens[1], m[1]));
        auto trigram =
            std::make_shared<v13::BitwiseXor>(std::make_shared<v13::BitwiseXor>(mul(tokens[0], m[0]), mul(extra, m[1])),
                                              mul(tokens[2], m[2]));
        auto concat =
            std::make_shared<v0::Concat>(OutputVector{block(bigram, 1000003, 0), block(trigram, 1000033, 1000003)}, -1);
        model = std::make_shared<Model>(OutputVector{concat}, tokens);
    }
    {
        auto tokens = make_tokens(kTrigram);
        auto extra = std::make_shared<v0::Parameter>(element::i64, kTrigram.token_shape);
        tokens.push_back(extra);
        const auto bigram = make_fused_order({tokens[0], tokens[1]}, head_range(m, 0, 2), {1000003}, {0});
        const auto trigram = make_fused_order({tokens[0], extra, tokens[2]}, m, {1000033}, {1000003});
        model_ref =
            std::make_shared<Model>(OutputVector{std::make_shared<v0::Concat>(OutputVector{bigram, trigram}, -1)},
                                    tokens);
    }
}

TEST_F(NGramHashFunctionFusionTests, DoesNotTouchUnrelatedI64AddAndConcat) {
    // Generic i64 Add / FloorMod + Add / Concat without the hashed XOR mix must not match.
    auto x = std::make_shared<v0::Parameter>(element::i64, PartialShape{-1, -1});
    auto y = std::make_shared<v0::Parameter>(element::i64, PartialShape{-1, -1});
    auto add = std::make_shared<v1::Add>(x, y);
    auto unsq = std::make_shared<v0::Unsqueeze>(add, v0::Constant::create(element::i64, Shape{}, {-1}));
    auto mod = std::make_shared<v1::FloorMod>(unsq, v0::Constant::create(element::i64, Shape{4}, {7, 11, 13, 17}));
    auto offset = std::make_shared<v1::Add>(mod, v0::Constant::create(element::i64, Shape{4}, {0, 7, 18, 31}));
    // A single Multiply term instead of an XOR mix.
    auto mul = std::make_shared<v1::Multiply>(x, v0::Constant::create(element::i64, Shape{}, {kWrapMultipliers[0]}));
    auto unsq_mul = std::make_shared<v0::Unsqueeze>(mul, v0::Constant::create(element::i64, Shape{}, {-1}));
    auto mod_mul = std::make_shared<v1::FloorMod>(unsq_mul, v0::Constant::create(element::i64, Shape{1}, {7}));
    auto offset_mul = std::make_shared<v1::Add>(mod_mul, v0::Constant::create(element::i64, Shape{1}, {0}));
    auto concat = std::make_shared<v0::Concat>(OutputVector{offset, offset_mul}, -1);
    model = std::make_shared<Model>(OutputVector{concat}, ParameterVector{x, y});
}

TEST_F(NGramHashFunctionFusionTests, DoesNotFuseXorWithNonConstantMultiplier) {
    const auto tokens = make_tokens(kTrigram);
    auto term0 = std::make_shared<v1::Multiply>(tokens[0], tokens[1]);
    auto term1 =
        std::make_shared<v1::Multiply>(tokens[1], v0::Constant::create(element::i64, Shape{}, {kWrapMultipliers[1]}));
    auto mix = std::make_shared<v13::BitwiseXor>(term0, term1);
    auto unsq = std::make_shared<v0::Unsqueeze>(mix, v0::Constant::create(element::i64, Shape{}, {-1}));
    auto mod = std::make_shared<v1::FloorMod>(unsq, v0::Constant::create(element::i64, Shape{1}, {1000003}));
    auto add = std::make_shared<v1::Add>(mod, v0::Constant::create(element::i64, Shape{1}, {0}));
    model = std::make_shared<Model>(OutputVector{add}, tokens);
}

TEST_F(NGramHashFunctionFusionTests, DoesNotDiscardUnprovenReshape) {
    model = make_hash_model(kBigram, make_tables(kBigram.heads_per_order));
    replace_unsqueeze_with_reshape(model, {-1, 2, 1}, false);
}

TEST_F(NGramHashFunctionFusionTests, DoesNotFuseInferredReshapeWithZeroInputDimension) {
    // Reshape([-1, 0, 1]) maps a [2, 0] input to [1, 0, 1], so removing it would change the output batch dimension.
    const HashGraphConfig config{"EmptyTokens", {11, 13}, {2}, {2, 0}};
    model = make_hash_model(config, make_tables(config.heads_per_order));
    replace_unsqueeze_with_reshape(model, {-1, 0, 1}, false);
    EXPECT_EQ(model->output(0).get_partial_shape(), PartialShape({1, 0, 2}));
}

TEST_F(NGramHashFunctionFusionTests, FusesProvenStaticReshapeAppend) {
    const HashGraphConfig config{"Reshape", {11, 13}, {2}, {2, 3}};
    const auto t = make_tables(config.heads_per_order);
    model = make_hash_model(config, t);
    replace_unsqueeze_with_reshape(model, {2, 3, 1}, false);
    model_ref = make_fused_model(config, t);
}

TEST_F(NGramHashFunctionFusionTests, FusesProvenSpecialZeroReshapeAppend) {
    const auto t = make_tables(kBigram.heads_per_order);
    model = make_hash_model(kBigram, t);
    replace_unsqueeze_with_reshape(model, {0, 0, 1}, true);
    model_ref = make_fused_model(kBigram, t);
}

TEST_F(NGramHashFunctionFusionTests, FusesSingleInferredPrefixDimension) {
    const HashGraphConfig config{"Reshape", {11, 13}, {2}, {-1}};
    const auto t = make_tables(config.heads_per_order);
    model = make_hash_model(config, t);
    replace_unsqueeze_with_reshape(model, {-1, 1}, false);
    model_ref = make_fused_model(config, t);
}

TEST_F(NGramHashFunctionFusionTests, FusesReshapeWithSingleDynamicDimension) {
    // NopElimination turns an Unsqueeze with one dynamic dimension into this Reshape.
    const HashGraphConfig config{"Reshape", {11, 13}, {2}, {-1, 3}};
    const auto t = make_tables(config.heads_per_order);
    model = make_hash_model(config, t);
    replace_unsqueeze_with_reshape(model, {-1, 3, 1}, false);
    model_ref = make_fused_model(config, t);
}

TEST_F(NGramHashFunctionFusionTests, DoesNotFuseReshapeWithDynamicAndZeroDimensions) {
    // The shapes match ([?, 0] -> [?, 0, 1]), but Reshape([-1, 0, 1]) maps a [2, 0] input to [0, 0, 1].
    const HashGraphConfig config{"EmptyTokens", {11, 13}, {2}, {-1, 0}};
    model = make_hash_model(config, make_tables(config.heads_per_order));
    replace_unsqueeze_with_reshape(model, {-1, 0, 1}, false);
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type<v1::Reshape>(node)) {
            EXPECT_EQ(node->get_output_partial_shape(0), PartialShape({-1, 0, 1}));
        }
    }
}

TEST_F(NGramHashFunctionFusionTests, DoesNotFuseReshapeThatSwapsDynamicDimensions) {
    // The shapes match ([?, ?] -> [?, ?, 1]), but the target shape swaps the two input dimensions.
    model = make_hash_model(kBigram, make_tables(kBigram.heads_per_order));
    for (const auto& node : model->get_ordered_ops()) {
        if (const auto unsqueeze = ov::as_type_ptr<v0::Unsqueeze>(node)) {
            const auto x = unsqueeze->input_value(0);
            const auto swapped = std::make_shared<v8::Gather>(std::make_shared<v3::ShapeOf>(x),
                                                              v0::Constant::create(element::i64, Shape{2}, {1, 0}),
                                                              v0::Constant::create(element::i64, Shape{}, {0}));
            const auto target =
                std::make_shared<v0::Concat>(OutputVector{swapped, v0::Constant::create(element::i64, Shape{1}, {1})},
                                             0);
            const auto reshape = std::make_shared<v1::Reshape>(x, target, false);
            EXPECT_EQ(reshape->get_output_partial_shape(0), PartialShape({-1, -1, 1}));
            ov::replace_node(unsqueeze, reshape);
        }
    }
    model->validate_nodes_and_infer_types();
}

// A single trigram block whose top XOR has the newest term on the given side; collect order follows the XOR inputs.
struct TrigramBlock {
    std::shared_ptr<Model> model;
    std::shared_ptr<Model> fused;
};

TrigramBlock make_trigram_block(const Shape& multiplier_shape,
                                bool term_first,
                                const PartialShape& token_shape = {-1, -1}) {
    const HashGraphConfig c{"TrigramBlock", {11, 13, 17}, {0, 4}, token_shape};
    const auto t = make_tables(c.heads_per_order);
    const auto tokens = make_tokens(c);
    auto term = [&](size_t k) {
        return std::make_shared<v1::Multiply>(tokens[k],
                                              v0::Constant::create(element::i64, multiplier_shape, {c.multipliers[k]}));
    };
    const auto inner = std::make_shared<v13::BitwiseXor>(term(0), term(1));
    const auto mix = term_first ? std::make_shared<v13::BitwiseXor>(term(2), inner)
                                : std::make_shared<v13::BitwiseXor>(inner, term(2));
    const auto unsq = std::make_shared<v0::Unsqueeze>(mix, v0::Constant::create(element::i64, Shape{}, {-1}));
    Shape vocab_shape(token_shape.size() + 1, 1);
    vocab_shape.back() = t.vocab_sizes.size();
    const auto mod =
        std::make_shared<v1::FloorMod>(unsq, v0::Constant::create(element::i64, vocab_shape, t.vocab_sizes));
    const auto add =
        std::make_shared<v1::Add>(mod, v0::Constant::create(element::i64, Shape{t.offsets.size()}, t.offsets));

    const std::vector<size_t> order = term_first ? std::vector<size_t>{2, 0, 1} : std::vector<size_t>{0, 1, 2};
    const auto ref_tokens = make_tokens(c);
    OutputVector fused_tokens;
    std::vector<int64_t> fused_multipliers;
    for (const auto k : order) {
        fused_tokens.push_back(ref_tokens[k]);
        fused_multipliers.push_back(c.multipliers[k]);
    }
    const auto fused = make_fused_order(fused_tokens, fused_multipliers, t.vocab_sizes, t.offsets);
    return {std::make_shared<Model>(OutputVector{add}, tokens),
            std::make_shared<Model>(OutputVector{fused}, ref_tokens)};
}

class NGramHashFunctionFusionXorOrderTests : public NGramHashFunctionFusionTests,
                                             public testing::WithParamInterface<bool> {};

// BitwiseXor is not commutative for the pattern matcher, so the term must be found on either XOR input.
TEST_P(NGramHashFunctionFusionXorOrderTests, FusesTermOnEitherXorInput) {
    auto block = make_trigram_block(Shape{}, GetParam());
    model = block.model;
    model_ref = block.fused;
    expect_same_outputs(model, model_ref);
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         NGramHashFunctionFusionXorOrderTests,
                         testing::Bool(),
                         [](const testing::TestParamInfo<bool>& info) {
                             return info.param ? std::string("TermFirst") : std::string("TermLast");
                         });

class NGramHashFunctionFusionMultiplierShapeTests : public NGramHashFunctionFusionTests,
                                                    public testing::WithParamInterface<Shape> {};

TEST_P(NGramHashFunctionFusionMultiplierShapeTests, FusesScalarLikeMultipliers) {
    auto block = make_trigram_block(GetParam(), false);
    model = block.model;
    model_ref = block.fused;
    expect_same_outputs(model, model_ref);
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         NGramHashFunctionFusionMultiplierShapeTests,
                         testing::Values(Shape{}, Shape{1}, Shape{1, 1}),
                         [](const testing::TestParamInfo<Shape>& info) {
                             return "Rank" + std::to_string(info.param.size());
                         });

TEST_F(NGramHashFunctionFusionTests, DoesNotFuseOtherMultiplierShape) {
    model = make_trigram_block(Shape{1, 1, 1}, false, {-1, -1, -1}).model;
}

// A [1, 1] multiplier broadcasts rank-1 tokens to rank 2, which packing the tokens would not.
TEST_F(NGramHashFunctionFusionTests, DoesNotFuseMultiplierThatRaisesDynamicTokenRank) {
    model = make_trigram_block(Shape{1, 1}, false, {-1}).model;
}

TEST_F(NGramHashFunctionFusionTests, DoesNotFuseMultiplierThatRaisesStaticTokenRank) {
    model = make_trigram_block(Shape{1, 1}, true, {5}).model;
}

class NGramHashFunctionFusionNonConstantMultiplierTests : public NGramHashFunctionFusionTests,
                                                          public testing::WithParamInterface<size_t> {};

// A scalar-shaped multiplier whose value isn't known when fusing, either on the top term matched by the pattern (2)
// or on a term found by the manual XOR walk (0).
TEST_P(NGramHashFunctionFusionNonConstantMultiplierTests, DoesNotFuseScalarParameterMultiplier) {
    const auto tokens = make_tokens(kTrigram);
    const auto multiplier = std::make_shared<v0::Parameter>(element::i64, PartialShape{});
    auto term = [&](size_t k) {
        const Output<Node> factor =
            k == GetParam() ? Output<Node>(multiplier)
                            : v0::Constant::create(element::i64, Shape{}, {kTrigram.multipliers[k]})->output(0);
        return std::make_shared<v1::Multiply>(tokens[k], factor);
    };
    const auto mix = std::make_shared<v13::BitwiseXor>(std::make_shared<v13::BitwiseXor>(term(0), term(1)), term(2));
    const auto unsq = std::make_shared<v0::Unsqueeze>(mix, v0::Constant::create(element::i64, Shape{}, {-1}));
    const auto mod = std::make_shared<v1::FloorMod>(unsq, v0::Constant::create(element::i64, Shape{1}, {1000003}));
    const auto add = std::make_shared<v1::Add>(mod, v0::Constant::create(element::i64, Shape{1}, {0}));
    auto params = tokens;
    params.push_back(multiplier);
    model = std::make_shared<Model>(OutputVector{add}, params);
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         NGramHashFunctionFusionNonConstantMultiplierTests,
                         testing::Values(size_t{2}, size_t{0}),
                         [](const testing::TestParamInfo<size_t>& info) {
                             return info.param == 2 ? std::string("MatchedTerm") : std::string("WalkedTerm");
                         });

class NGramHashFunctionFusionNonNumpyTests : public NGramHashFunctionFusionTests,
                                             public testing::WithParamInterface<std::string> {};

TEST_P(NGramHashFunctionFusionNonNumpyTests, DoesNotFuseNonNumpyHeadBroadcasts) {
    const HashGraphConfig config{"Broadcast", {11, 13}, {2}, {2}};
    model = make_hash_model(config, make_tables(config.heads_per_order));
    for (const auto& node : model->get_ordered_ops()) {
        if (node->get_type_name() == GetParam()) {
            const auto arithmetic = ov::as_type_ptr<ov::op::util::BinaryElementwiseArithmetic>(node);
            const auto constant = ov::as_type_ptr<v0::Constant>(arithmetic->get_input_node_shared_ptr(1));
            arithmetic->input(1).replace_source_output(
                v0::Constant::create(element::i64, Shape{2}, constant->cast_vector<int64_t>()));
            arithmetic->set_autob(ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::PDPD, 0));
        }
    }
    model->validate_nodes_and_infer_types();
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         NGramHashFunctionFusionNonNumpyTests,
                         testing::Values("FloorMod", "Add"),
                         [](const testing::TestParamInfo<std::string>& info) {
                             return info.param;
                         });

TEST_F(NGramHashFunctionFusionTests, DoesNotFuseStaticallyBroadcastTokens) {
    const HashGraphConfig config{"Bigram", {11, 13}, {2}, {3}};
    model = make_hash_model(config, make_tables(config.heads_per_order));
    model->get_parameters()[0]->set_partial_shape({1});
    model->validate_nodes_and_infer_types();
}

TEST_F(NGramHashFunctionFusionTests, FusesInt32RowBoundary) {
    const HashGraphConfig config{"RowBoundary", {11, 13}, {1}, {2}};
    const HashTables t{{2147483648LL}, {0}};
    model = make_hash_model(config, t);
    model_ref = make_fused_model(config, t);
}

TEST_F(NGramHashFunctionFusionTests, DoesNotMergeExistingHashWithRepeatedTerm) {
    // x * 11 ^ x * 11 cancels out; merging it as a set of terms would change the result.
    const auto x = std::make_shared<v0::Parameter>(element::i64, PartialShape{-1, -1});
    const auto y = std::make_shared<v0::Parameter>(element::i64, PartialShape{-1, -1});
    model = std::make_shared<Model>(OutputVector{std::make_shared<v0::Concat>(
                                        OutputVector{make_fused_order({x, x}, {11, 11}, {1000003}, {0}),
                                                     make_fused_order({x, y, y}, {11, 13, 17}, {1000033}, {1000003})},
                                        -1)},
                                    ParameterVector{x, y});
}

TEST_F(NGramHashFunctionFusionTests, CallbackSkipsBlockFusion) {
    model = make_hash_model(kTrigram, make_tables(kTrigram.heads_per_order));
    manager.get_pass_config()->set_callback<ov::pass::NGramHashBlockFusion>([](const std::shared_ptr<const Node>&) {
        return true;
    });
}

TEST_F(NGramHashFunctionFusionTests, CallbackSkipsConcatFusion) {
    const auto t = make_tables(kTrigram.heads_per_order);
    model = make_hash_model(kTrigram, t);
    manager.get_pass_config()->set_callback<ov::pass::NGramHashConcatFusion>([](const std::shared_ptr<const Node>&) {
        return true;
    });
    const auto tokens = make_tokens(kTrigram);
    const auto& m = kTrigram.multipliers;
    const auto bigram = make_fused_order({tokens[0], tokens[1]},
                                         head_range(m, 0, 2),
                                         head_range(t.vocab_sizes, 0, 4),
                                         head_range(t.offsets, 0, 4));
    const auto trigram = make_fused_order({tokens[0], tokens[1], tokens[2]},
                                          m,
                                          head_range(t.vocab_sizes, 4, 8),
                                          head_range(t.offsets, 4, 8));
    model_ref =
        std::make_shared<Model>(OutputVector{std::make_shared<v0::Concat>(OutputVector{bigram, trigram}, -1)}, tokens);
}

class NGramHashFunctionFusionAfterNopEliminationTests : public TransformationTestsF {
public:
    NGramHashFunctionFusionAfterNopEliminationTests() {
        comparator.enable(FunctionsComparator::CONST_VALUES);
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<ov::pass::NopElimination>();
        manager.register_pass<ov::pass::NGramHashFunctionFusion>();
    }
};

TEST_F(NGramHashFunctionFusionAfterNopEliminationTests, FusesBlocksWithRemovedZeroOffsetsAdd) {
    auto t = make_tables(kTrigram.heads_per_order);
    std::fill(t.offsets.begin(), t.offsets.end(), 0);
    model = make_hash_model(kTrigram, t);
    model_ref = make_fused_model(kTrigram, t);
}
