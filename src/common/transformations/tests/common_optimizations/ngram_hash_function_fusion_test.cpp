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
#include "openvino/op/multiply.hpp"
#include "openvino/op/ngram_hash_function.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/manager.hpp"

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

size_t count_ops(const std::shared_ptr<Model>& model, const std::string& type_name) {
    size_t count = 0;
    for (const auto& op : model->get_ops()) {
        count += std::string(op->get_type_name()) == type_name;
    }
    return count;
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

}  // namespace

class NGramHashFunctionFusionTest : public testing::Test, public testing::WithParamInterface<HashGraphConfig> {
public:
    static std::string get_test_case_name(const testing::TestParamInfo<HashGraphConfig>& info) {
        return info.param.name;
    }
};

TEST_P(NGramHashFunctionFusionTest, FusesAllOrdersIntoOneOp) {
    const auto& c = GetParam();
    const auto t = make_tables(c.heads_per_order);
    const auto model = make_hash_model(c, t);
    const auto original = model->clone();
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::NGramHashFunctionFusion>();
    manager.run_passes(model);
    ASSERT_EQ(count_ops(model, "NGramHashFunction"), 1);
    for (const auto& node : model->get_ops()) {
        if (const auto hash = ov::as_type_ptr<ov::op::internal::NGramHashFunction>(node)) {
            EXPECT_EQ(hash->get_heads_per_order(), c.heads_per_order);
            EXPECT_EQ(hash->get_head_vocab_sizes(), t.vocab_sizes);
            EXPECT_EQ(hash->get_head_offsets(), t.offsets);
        }
    }
    ov::TensorVector inputs;
    for (const auto& parameter : original->get_parameters()) {
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
    const auto actual = ov::test::utils::infer_on_template(model, inputs);
    auto expected_shape = inputs[0].get_shape();
    expected_shape.push_back(t.vocab_sizes.size());
    Tensor expected(element::i64, expected_shape);
    for (size_t position = 0; position < inputs[0].get_size(); ++position) {
        uint64_t mix = 0;
        size_t head = 0;
        for (size_t back = 0; back < inputs.size(); ++back) {
            mix ^= static_cast<uint64_t>(inputs[back].data<int64_t>()[position]) *
                   static_cast<uint64_t>(c.multipliers[back]);
            if (back == 0) {
                continue;
            }
            const auto signed_mix = static_cast<int64_t>(mix);
            for (int64_t h = 0; h < c.heads_per_order[back - 1]; ++h, ++head) {
                auto remainder = signed_mix % t.vocab_sizes[head];
                if (remainder < 0) {
                    remainder += t.vocab_sizes[head];
                }
                expected.data<int64_t>()[position * t.vocab_sizes.size() + head] = remainder + t.offsets[head];
            }
        }
    }
    ASSERT_EQ(actual[0].get_shape(), expected.get_shape());
    EXPECT_EQ(std::vector<int64_t>(actual[0].data<int64_t>(), actual[0].data<int64_t>() + actual[0].get_size()),
              std::vector<int64_t>(expected.data<int64_t>(), expected.data<int64_t>() + expected.get_size()));
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         NGramHashFunctionFusionTest,
                         testing::ValuesIn(kPositiveCases),
                         NGramHashFunctionFusionTest::get_test_case_name);

namespace {
void expect_not_fused(const std::shared_ptr<Model>& model) {
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::NGramHashFunctionFusion>();
    manager.run_passes(model);
    EXPECT_EQ(count_ops(model, "NGramHashFunction"), 0);
}
const HashGraphConfig kTrigram{"Trigram", {kWrapMultipliers.begin(), kWrapMultipliers.begin() + 3}, {4, 4}, {-1, -1}};
}  // namespace

TEST(NGramHashFunctionFusionNegativeTest, DoesNotFuseI32) {
    const HashGraphConfig i32_trigram{"I32Trigram", {1000003, 999983, 7}, {4, 4}, {-1, -1}};
    expect_not_fused(make_hash_model(i32_trigram, make_tables(i32_trigram.heads_per_order, 1000003), element::i32));
}

TEST(NGramHashFunctionFusionNegativeTest, DoesNotFuseWhenRowsOverflowInt32) {
    auto t = make_tables(kTrigram.heads_per_order);
    t.offsets.back() = std::numeric_limits<int32_t>::max() - t.vocab_sizes.back() + 2;
    const auto model = make_hash_model(kTrigram, t);
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::NGramHashFunctionFusion>();
    manager.run_passes(model);
    // The Concat is not fused; only the bigram block, whose rows fit int32, becomes an op.
    ASSERT_EQ(count_ops(model, "NGramHashFunction"), 1);
    EXPECT_EQ(count_ops(model, "FloorMod"), 1);
    for (const auto& op : model->get_ops()) {
        if (const auto hash = ov::as_type_ptr<ov::op::internal::NGramHashFunction>(op)) {
            EXPECT_EQ(hash->get_ngram_size(), 2);
            EXPECT_EQ(hash->get_heads_per_order(), std::vector<int64_t>{4});
        }
    }
}

TEST(NGramHashFunctionFusionNegativeTest, DoesNotFuseUnsqueezeOnOtherAxis) {
    expect_not_fused(make_hash_model(kTrigram, make_tables(kTrigram.heads_per_order), element::i64, 0));
}

TEST(NGramHashFunctionFusionNegativeTest, FusesEveryOrderOnItsOwnWhenConcatOrderIsDescending) {
    // Heads must be grouped by ascending order to fit one op; otherwise every block is fused separately.
    const auto model = make_hash_model(kTrigram, make_tables(kTrigram.heads_per_order), element::i64, -1, true);
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::NGramHashFunctionFusion>();
    manager.run_passes(model);
    EXPECT_EQ(count_ops(model, "NGramHashFunction"), 2);
}

TEST(NGramHashFunctionFusionNegativeTest, DoesNotFuseWhenHigherOrderDoesNotExtendLowerOrder) {
    // The trigram uses a different second token than the bigram: not one n-gram window.
    auto tokens = make_tokens(kTrigram);
    auto extra = std::make_shared<v0::Parameter>(element::i64, kTrigram.token_shape);
    tokens.push_back(extra);
    auto mul = [](const Output<Node>& x, int64_t m) {
        return std::make_shared<v1::Multiply>(x, v0::Constant::create(element::i64, Shape{}, {m}));
    };
    auto block = [](const Output<Node>& mix, int64_t vocab, int64_t offset) {
        auto unsq = std::make_shared<v0::Unsqueeze>(mix, v0::Constant::create(element::i64, Shape{}, {-1}));
        auto mod = std::make_shared<v1::FloorMod>(unsq, v0::Constant::create(element::i64, Shape{1}, {vocab}));
        return std::make_shared<v1::Add>(mod, v0::Constant::create(element::i64, Shape{1}, {offset}));
    };
    const auto& m = kTrigram.multipliers;
    auto bigram = std::make_shared<v13::BitwiseXor>(mul(tokens[0], m[0]), mul(tokens[1], m[1]));
    auto trigram = std::make_shared<v13::BitwiseXor>(
        std::make_shared<v13::BitwiseXor>(mul(tokens[0], m[0]), mul(extra, m[1])),
        mul(tokens[2], m[2]));
    auto concat = std::make_shared<v0::Concat>(OutputVector{block(bigram, 1000003, 0), block(trigram, 1000033, 1000003)},
                                               -1);
    const auto model = std::make_shared<Model>(OutputVector{concat}, tokens);
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::NGramHashFunctionFusion>();
    manager.run_passes(model);
    // Not one op for the Concat; the blocks are fused on their own.
    EXPECT_EQ(count_ops(model, "NGramHashFunction"), 2);
}

TEST(NGramHashFunctionFusionNegativeTest, DoesNotTouchUnrelatedI64AddAndConcat) {
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
    const auto model = std::make_shared<Model>(OutputVector{concat}, ParameterVector{x, y});
    expect_not_fused(model);
    EXPECT_EQ(count_ops(model, "Add"), 3);
    EXPECT_EQ(count_ops(model, "Concat"), 1);
}

TEST(NGramHashFunctionFusionNegativeTest, DoesNotFuseXorWithNonConstantMultiplier) {
    const auto tokens = make_tokens(kTrigram);
    auto term0 = std::make_shared<v1::Multiply>(tokens[0], tokens[1]);
    auto term1 =
        std::make_shared<v1::Multiply>(tokens[1], v0::Constant::create(element::i64, Shape{}, {kWrapMultipliers[1]}));
    auto mix = std::make_shared<v13::BitwiseXor>(term0, term1);
    auto unsq = std::make_shared<v0::Unsqueeze>(mix, v0::Constant::create(element::i64, Shape{}, {-1}));
    auto mod = std::make_shared<v1::FloorMod>(unsq, v0::Constant::create(element::i64, Shape{1}, {1000003}));
    auto add = std::make_shared<v1::Add>(mod, v0::Constant::create(element::i64, Shape{1}, {0}));
    expect_not_fused(std::make_shared<Model>(OutputVector{add}, tokens));
}

TEST(NGramHashFunctionFusionNegativeTest, DoesNotDiscardUnprovenReshape) {
    const HashGraphConfig config{"Reshape", {11, 13}, {2}, {-1, -1}};
    const auto model = make_hash_model(config, make_tables(config.heads_per_order));
    for (const auto& node : model->get_ordered_ops()) {
        if (const auto unsqueeze = ov::as_type_ptr<v0::Unsqueeze>(node)) {
            ov::replace_node(unsqueeze,
                             std::make_shared<v1::Reshape>(unsqueeze->input_value(0),
                                                           v0::Constant::create(element::i64, Shape{3}, {-1, 2, 1}),
                                                           false));
        }
    }
    expect_not_fused(model);
}

TEST(NGramHashFunctionFusionRegressionTest, FusesProvenReshapeAppend) {
    for (const bool special_zero : {false, true}) {
        const HashGraphConfig config{"Reshape",
                                     {11, 13},
                                     {2},
                                     special_zero ? PartialShape{-1, -1} : PartialShape{2, 3}};
        const auto model = make_hash_model(config, make_tables(config.heads_per_order));
        for (const auto& node : model->get_ordered_ops()) {
            if (const auto unsqueeze = ov::as_type_ptr<v0::Unsqueeze>(node)) {
                const std::vector<int64_t> shape =
                    special_zero ? std::vector<int64_t>{0, 0, 1} : std::vector<int64_t>{2, 3, 1};
                ov::replace_node(unsqueeze,
                                 std::make_shared<v1::Reshape>(unsqueeze->input_value(0),
                                                               v0::Constant::create(element::i64, Shape{3}, shape),
                                                               special_zero));
            }
        }
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::NGramHashFunctionFusion>();
        manager.run_passes(model);
        EXPECT_EQ(count_ops(model, "NGramHashFunction"), 1);
    }
}

TEST(NGramHashFunctionFusionRegressionTest, FusesSingleInferredPrefixDimension) {
    const HashGraphConfig config{"Reshape", {11, 13}, {2}, {-1}};
    const auto model = make_hash_model(config, make_tables(config.heads_per_order));
    for (const auto& node : model->get_ordered_ops()) {
        if (const auto unsqueeze = ov::as_type_ptr<v0::Unsqueeze>(node)) {
            ov::replace_node(unsqueeze,
                             std::make_shared<v1::Reshape>(unsqueeze->input_value(0),
                                                           v0::Constant::create(element::i64, Shape{2}, {-1, 1}),
                                                           false));
        }
    }
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::NGramHashFunctionFusion>();
    manager.run_passes(model);
    EXPECT_EQ(count_ops(model, "NGramHashFunction"), 1);
}

TEST(NGramHashFunctionFusionNegativeTest, DoesNotFuseNonNumpyHeadBroadcasts) {
    const HashGraphConfig config{"Broadcast", {11, 13}, {2}, {2}};
    for (const bool change_add : {false, true}) {
        const auto model = make_hash_model(config, make_tables(config.heads_per_order));
        for (const auto& node : model->get_ordered_ops()) {
            if ((change_add && ov::is_type<v1::Add>(node)) || (!change_add && ov::is_type<v1::FloorMod>(node))) {
                const auto arithmetic = ov::as_type_ptr<ov::op::util::BinaryElementwiseArithmetic>(node);
                const auto constant = ov::as_type_ptr<v0::Constant>(arithmetic->get_input_node_shared_ptr(1));
                arithmetic->input(1).replace_source_output(
                    v0::Constant::create(element::i64, Shape{2}, constant->cast_vector<int64_t>()));
                arithmetic->set_autob(ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::PDPD, 0));
            }
        }
        model->validate_nodes_and_infer_types();
        expect_not_fused(model);
    }
}

TEST(NGramHashFunctionFusionNegativeTest, DoesNotFuseStaticallyBroadcastTokens) {
    const HashGraphConfig config{"Bigram", {11, 13}, {2}, {3}};
    const auto model = make_hash_model(config, make_tables(config.heads_per_order));
    model->get_parameters()[0]->set_partial_shape({1});
    model->validate_nodes_and_infer_types();
    expect_not_fused(model);
}

TEST(NGramHashFunctionFusionRegressionTest, FusesInt32RowBoundary) {
    const HashGraphConfig config{"RowBoundary", {11, 13}, {1}, {2}};
    const auto model = make_hash_model(config, {{2147483648LL}, {0}});
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::NGramHashFunctionFusion>();
    EXPECT_NO_THROW(manager.run_passes(model));
    EXPECT_EQ(count_ops(model, "NGramHashFunction"), 1);
}
