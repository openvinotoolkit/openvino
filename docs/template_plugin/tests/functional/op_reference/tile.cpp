// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/opsets/opset1.hpp"
#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct TileParams {
    TileParams(
        const Tensor& A, const Tensor& repeats,
        const Tensor& expected, const std::string& testcaseName = "") :
        A(A), repeats(repeats), expected(expected), testcaseName(testcaseName) {}

    Tensor A;
    Tensor repeats;
    Tensor expected;
    std::string testcaseName;
};

class ReferenceTileTest : public testing::TestWithParam<TileParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.A.data};
        refOutData = {params.expected.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<TileParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "aType=" << param.A.type;
        result << "_aShape=" << param.A.shape;
        result << "_rType=" << param.repeats.type;
        result << "_rShape=" << param.repeats.shape;
        result << "_eType=" << param.expected.type;
        if (param.testcaseName != "") {
            result << "_eShape=" << param.expected.shape;
            result << "_=" << param.testcaseName;
        } else {
            result << "_rShape=" << param.expected.shape;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const TileParams& params) {
        const auto A = std::make_shared<opset1::Parameter>(params.A.type, params.A.shape);
        const auto repeats = std::make_shared<opset1::Constant>(params.repeats.type, params.repeats.shape,
                                                                params.repeats.data.data());
        const auto tile = std::make_shared<opset1::Tile>(A, repeats);
        const auto f = std::make_shared<Function>(NodeVector{tile}, ParameterVector{A});
        return f;
    }
};

TEST_P(ReferenceTileTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<TileParams> generateParams() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<TileParams> params {
        TileParams(
            Tensor(ET, {3}, std::vector<T>{1, 2, 3}),
            Tensor(element::i64, {3}, std::vector<int64_t>{2, 2, 1}),
            Tensor(ET, {2, 2, 3}, std::vector<T>{1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}),
            "tile_3d_small_data_rank"),
        TileParams(
            Tensor(ET, {2, 1, 3}, std::vector<T>{1, 2, 3, 4, 5, 6}),
            Tensor(element::i64, {2}, std::vector<int64_t>{2, 1}),
            Tensor(ET, {2, 2, 3}, std::vector<T>{1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6}),
            "tile_3d_few_repeats"),
    };
    return params;
}

std::vector<TileParams> generateCombinedParams() {
    const std::vector<std::vector<TileParams>> generatedParams {
        generateParams<element::Type_t::i8>(),
        generateParams<element::Type_t::i16>(),
        generateParams<element::Type_t::i32>(),
        generateParams<element::Type_t::i64>(),
        generateParams<element::Type_t::u8>(),
        generateParams<element::Type_t::u16>(),
        generateParams<element::Type_t::u32>(),
        generateParams<element::Type_t::u64>(),
        generateParams<element::Type_t::bf16>(),
        generateParams<element::Type_t::f16>(),
        generateParams<element::Type_t::f32>(),
        generateParams<element::Type_t::f64>(),
    };
    std::vector<TileParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Tile_With_Hardcoded_Refs, ReferenceTileTest,
    testing::ValuesIn(generateCombinedParams()), ReferenceTileTest::getTestCaseName);
} // namespace