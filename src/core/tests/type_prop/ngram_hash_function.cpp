// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/ngram_hash_function.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/parameter.hpp"

namespace ov {
namespace test {

using ov::op::internal::NGramHashFunction;
using ov::op::v0::Parameter;
using testing::HasSubstr;

namespace {
// Qwen3.8-Flash-Next style: trigram, 8 heads per order (vocab sizes shortened).
const std::vector<int64_t> kQwenMultipliers{23703573157769, 20109073645365, 8052911324071};
struct HashConfig {
    std::vector<int64_t> heads_per_order;
    std::vector<int64_t> multipliers;
    std::vector<int64_t> vocab_sizes;
    std::vector<int64_t> offsets;
};

HashConfig make_config(std::vector<int64_t> heads_per_order, std::vector<int64_t> multipliers) {
    HashConfig c{std::move(heads_per_order), std::move(multipliers), {}, {}};
    int64_t offset = 0;
    int64_t vocab = 1000003;
    for (const auto heads : c.heads_per_order) {
        for (int64_t e = 0; e < heads; ++e) {
            c.vocab_sizes.push_back(vocab);
            c.offsets.push_back(offset);
            offset += vocab;
            vocab += 30;
        }
    }
    return c;
}

std::shared_ptr<NGramHashFunction> make_hash(const Output<Node>& ngrams, const HashConfig& c) {
    return std::make_shared<NGramHashFunction>(ngrams, c.heads_per_order, c.multipliers, c.vocab_sizes, c.offsets);
}

}  // namespace

class TypePropNGramHashFunctionTest : public TypePropOpTest<NGramHashFunction> {};

TEST_F(TypePropNGramHashFunctionTest, qwen_trigram) {
    const auto c = make_config({8, 8}, kQwenMultipliers);
    const auto op = make_hash(std::make_shared<Parameter>(element::i64, PartialShape{2, 5, 3}), c);

    EXPECT_EQ(op->get_input_size(), 1);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 5, 16}));
    EXPECT_EQ(op->get_num_heads(), 16);
}

TEST_F(TypePropNGramHashFunctionTest, shapes_sdpa_paged_and_dynamic) {
    const auto c = make_config({8, 8}, kQwenMultipliers);
    for (const auto& [in, out] :
         std::vector<std::pair<PartialShape, PartialShape>>{{{1, 300, 3}, {1, 300, 16}},
                                                            {{-1, -1, 3}, {-1, -1, 16}},
                                                            {{-1, 3}, {-1, 16}},  // continuous batching [T, ngram_size]
                                                            {{-1, -1}, {-1, 16}},
                                                            {{3}, {16}},
                                                            {PartialShape::dynamic(), PartialShape::dynamic()}}) {
        for (const auto type : {element::i32, element::i64}) {
            const auto op = make_hash(std::make_shared<Parameter>(type, in), c);
            EXPECT_EQ(op->get_output_element_type(0), element::i32);
            EXPECT_EQ(op->get_output_partial_shape(0), out) << in;
        }
    }
}

TEST_F(TypePropNGramHashFunctionTest, per_order_heads) {
    const auto c = make_config({4, 2, 6}, {11, 13, 17, 19});
    const auto op = make_hash(std::make_shared<Parameter>(element::i64, PartialShape{-1, 4}), c);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, 12}));
}

TEST_F(TypePropNGramHashFunctionTest, invalid_inputs_and_attributes) {
    const auto c = make_config({8, 8}, kQwenMultipliers);
    const auto ngrams = std::make_shared<Parameter>(element::i64, PartialShape{-1, 3});

    OV_EXPECT_THROW(make_hash(std::make_shared<Parameter>(element::i64, PartialShape{-1, 4}), c),
                    NodeValidationFailure,
                    HasSubstr("last dimension of the ngrams input must be equal to ngram_size"));
    OV_EXPECT_THROW(make_hash(std::make_shared<Parameter>(element::f32, PartialShape{-1, 3}), c),
                    NodeValidationFailure,
                    HasSubstr("must be i32 or i64"));
    OV_EXPECT_THROW(make_hash(ngrams, {{}, {1}, {1}, {0}}),
                    NodeValidationFailure,
                    HasSubstr("multipliers must have ngram_size >= 2 values"));

    auto bad = c;
    bad.multipliers.pop_back();
    OV_EXPECT_THROW(make_hash(ngrams, bad), NodeValidationFailure, HasSubstr("heads_per_order must have"));

    bad = c;
    bad.heads_per_order = {8};
    OV_EXPECT_THROW(make_hash(ngrams, bad), NodeValidationFailure, HasSubstr("heads_per_order must have"));

    bad = c;
    bad.offsets.pop_back();
    OV_EXPECT_THROW(make_hash(ngrams, bad), NodeValidationFailure, HasSubstr("head_vocab_sizes and head_offsets"));

    bad = c;
    bad.vocab_sizes[3] = 0;
    OV_EXPECT_THROW(make_hash(ngrams, bad), NodeValidationFailure, HasSubstr("must be positive"));

    bad = c;
    bad.offsets[3] = -1;
    OV_EXPECT_THROW(make_hash(ngrams, bad), NodeValidationFailure, HasSubstr("must be non-negative"));

    bad = c;
    bad.offsets.back() = std::numeric_limits<int32_t>::max() - bad.vocab_sizes.back() + 2;
    OV_EXPECT_THROW(make_hash(ngrams, bad), NodeValidationFailure, HasSubstr("must fit int32"));
}

TEST_F(TypePropNGramHashFunctionTest, rejects_overflowing_head_totals) {
    const auto input = std::make_shared<Parameter>(element::i64, PartialShape{-1, 4});
    const auto max = std::numeric_limits<int64_t>::max();
    for (const auto& heads : std::vector<std::vector<int64_t>>{{max, max, 3}, {max, 1, 0}}) {
        OV_EXPECT_THROW(make_hash(input, {heads, {1, 2, 3, 4}, {7}, {0}}),
                        NodeValidationFailure,
                        HasSubstr("sum(heads_per_order)"));
    }
}

TEST_F(TypePropNGramHashFunctionTest, rejects_negative_and_zero_head_totals) {
    const auto input = std::make_shared<Parameter>(element::i64, PartialShape{-1, 3});
    OV_EXPECT_THROW(make_hash(input, {{-1, 1}, {1, 2, 3}, {7}, {0}}), NodeValidationFailure, HasSubstr("non-negative"));
    OV_EXPECT_THROW(make_hash(input, {{0, 0}, {1, 2, 3}, {}, {}}),
                    NodeValidationFailure,
                    HasSubstr("at least one head"));
}

TEST_F(TypePropNGramHashFunctionTest, accepts_largest_int32_row) {
    const auto input = std::make_shared<Parameter>(element::i64, PartialShape{-1, 2});
    const auto max = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    EXPECT_NO_THROW(make_hash(input, {{1}, {11, 13}, {max + 1}, {0}}));
    EXPECT_NO_THROW(make_hash(input, {{1}, {11, 13}, {1}, {max}}));
    for (const auto& [vocab, offset] :
         std::vector<std::pair<int64_t, int64_t>>{{max + 2, 0}, {max + 1, 1}, {1, max + 1}}) {
        OV_EXPECT_THROW(make_hash(input, {{1}, {11, 13}, {vocab}, {offset}}),
                        NodeValidationFailure,
                        HasSubstr("must fit int32"));
    }
}

}  // namespace test
}  // namespace ov
