// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "base_reference_test.hpp"
#include "openvino/op/ngram_hash_function.hpp"
#include "openvino/op/parameter.hpp"

namespace {
struct NGramHashFunctionParams {
    std::string name;
    reference_tests::Tensor input;
    std::vector<int64_t> heads_per_order;
    std::vector<int64_t> multipliers;
    std::vector<int64_t> vocab_sizes;
    std::vector<int64_t> offsets;
    reference_tests::Tensor expected;
};

class ReferenceNGramHashFunctionTest : public testing::TestWithParam<NGramHashFunctionParams>,
                                       public reference_tests::CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        const auto input = std::make_shared<ov::op::v0::Parameter>(params.input.type, params.input.shape);
        const auto hash = std::make_shared<ov::op::internal::NGramHashFunction>(
            input,
            static_cast<int64_t>(params.multipliers.size()),
            params.heads_per_order,
            params.multipliers,
            params.vocab_sizes,
            params.offsets);
        function = std::make_shared<ov::Model>(ov::OutputVector{hash}, ov::ParameterVector{input});
        inputData = {params.input.data};
        refOutData = {params.expected.data};
        threshold = 0;
        abs_threshold = 0;
    }
};

TEST_P(ReferenceNGramHashFunctionTest, CompareWithHardcodedRefs) {
    Exec();
}

std::vector<NGramHashFunctionParams> generate_params() {
    std::vector<NGramHashFunctionParams> params;
    for (const auto type : {ov::element::i32, ov::element::i64}) {
        const auto add_case = [&](const std::string& name,
                                  const ov::Shape& shape,
                                  const std::vector<int64_t>& tokens,
                                  const std::vector<int64_t>& heads,
                                  const std::vector<int64_t>& multipliers,
                                  const std::vector<int64_t>& vocab,
                                  const std::vector<int64_t>& offsets,
                                  const std::vector<int32_t>& expected) {
            const auto input = type == ov::element::i32
                                   ? reference_tests::Tensor{shape, type, std::vector<int32_t>(tokens.begin(), tokens.end())}
                                   : reference_tests::Tensor{shape, type, tokens};
            auto output_shape = shape;
            output_shape.back() = vocab.size();
            params.push_back({name + "_" + type.get_type_name(),
                              input,
                              heads,
                              multipliers,
                              vocab,
                              offsets,
                              reference_tests::Tensor{output_shape, ov::element::i32, expected}});
        };
        add_case("Bigram", {2}, {2, 3}, {2}, {3, 5}, {7, 11}, {0, 7}, {3, 10});
        add_case("TrigramSigned",
                 {2, 3},
                 {1, 2, 3, -1, -2, -3},
                 {2, 1},
                 {11, 13, 17},
                 {7, 11, 13},
                 {0, 7, 18},
                 {3, 11, 21, 1, 9, 28});
        add_case("WrappingProducts",
                 {1, 2, 3},
                 {-2147483648LL, 2147483647, -1, 0, 2000000000, -2000000000},
                 {1, 2},
                 {0x3BF58476D1CE4E5B, static_cast<int64_t>(0x9E3779B97F4A7C15ULL), 1234567891011},
                 {1000003, 1000033, 1},
                 {0, 1000003, 2147483647},
                 {696524, 1505995, 2147483647, 119144, 1949394, 2147483647});
        add_case("FourgramPerOrderHeads",
                 {1, 4},
                 {1, 2, 3, 4},
                 {2, 1, 2},
                 {11, 13, 17, 19},
                 {7, 11, 13, 17, 19},
                 {0, 7, 18, 31, 48},
                 {4, 7, 20, 38, 49});
        add_case("FivegramSkippedOrder",
                 {1, 5},
                 {1, 2, 3, 4, 5},
                 {1, 0, 1, 2},
                 {11, 13, 17, 19, 23},
                 {7, 11, 13, 17},
                 {0, 7, 18, 31},
                 {3, 7, 19, 32});
        add_case("ZeroTokens", {1, 3}, {0, 0, 0}, {1, 1}, {11, 13, 17}, {7, 11}, {0, 7}, {0, 7});
        add_case("Empty", {0, 3}, {}, {1, 1}, {11, 13, 17}, {7, 11}, {0, 7}, {});
        if (type == ov::element::i64) {
            add_case("FullInt64Range",
                     {1, 3},
                     {std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max(), 3000000000LL},
                     {1, 1},
                     {23703573157769, 20109073645365, 8052911324071},
                     {1000003, 1000033},
                     {0, 1000003},
                     {685983, 1927332});
        }
    }
    return params;
}

INSTANTIATE_TEST_SUITE_P(smoke_NGramHashFunction,
                         ReferenceNGramHashFunctionTest,
                         testing::ValuesIn(generate_params()),
                         [](const testing::TestParamInfo<NGramHashFunctionParams>& info) {
                             return info.param.name;
                         });
}  // namespace