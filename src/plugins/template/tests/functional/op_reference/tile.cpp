// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/tile.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct TileParams {
    TileParams(const reference_tests::Tensor& A,
               const reference_tests::Tensor& repeats,
               const reference_tests::Tensor& expected,
               const std::string& testcaseName = "")
        : A(A),
          repeats(repeats),
          expected(expected),
          testcaseName(testcaseName) {}

    reference_tests::Tensor A;
    reference_tests::Tensor repeats;
    reference_tests::Tensor expected;
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
    static std::shared_ptr<Model> CreateFunction(const TileParams& params) {
        const auto A = std::make_shared<op::v0::Parameter>(params.A.type, params.A.shape);
        const auto repeats =
            std::make_shared<op::v0::Constant>(params.repeats.type, params.repeats.shape, params.repeats.data.data());
        const auto tile = std::make_shared<op::v0::Tile>(A, repeats);
        const auto f = std::make_shared<Model>(NodeVector{tile}, ParameterVector{A});
        return f;
    }
};

TEST_P(ReferenceTileTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET, element::Type_t ET_INT>
std::vector<TileParams> generateParams() {
    using T = typename element_type_traits<ET>::value_type;
    using T_INT = typename element_type_traits<ET_INT>::value_type;
    std::vector<TileParams> params{
        TileParams(reference_tests::Tensor(ET, {}, std::vector<T>{1}),
                   reference_tests::Tensor(ET_INT, {1}, std::vector<T_INT>{2}),
                   reference_tests::Tensor(ET, {2}, std::vector<T>{1, 1}),
                   "tile_0d_to_1d_data_broadcast"),
        TileParams(reference_tests::Tensor(ET,
                                           {6},
                                           std::vector<T>{
                                               1,
                                               2,
                                               3,
                                               4,
                                               5,
                                               6,
                                           }),
                   reference_tests::Tensor(ET_INT, {1}, std::vector<T_INT>{4}),
                   reference_tests::Tensor(ET, {24}, std::vector<T>{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6,
                                                                    1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6}),
                   "tile_1d_to_1d_no_broadcast"),
        TileParams(reference_tests::Tensor(ET, {3}, std::vector<T>{1, 2, 3}),
                   reference_tests::Tensor(ET_INT, {3}, std::vector<T_INT>{2, 2, 1}),
                   reference_tests::Tensor(ET, {2, 2, 3}, std::vector<T>{1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}),
                   "tile_1d_to_3d_data_broadcast"),
        TileParams(reference_tests::Tensor(ET, {2, 1, 3}, std::vector<T>{1, 2, 3, 4, 5, 6}),
                   reference_tests::Tensor(ET_INT, {2}, std::vector<T_INT>{2, 1}),
                   reference_tests::Tensor(ET, {2, 2, 3}, std::vector<T>{1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6}),
                   "tile_3d_to_3d_repeats_broadcast"),
        TileParams(reference_tests::Tensor(ET, {1}, std::vector<T>{1}),
                   reference_tests::Tensor(ET_INT, {3}, std::vector<T_INT>{0, 2, 3}),
                   reference_tests::Tensor(ET, {0, 2, 3}, std::vector<T>{}),
                   "tile_1d_to_3d_with_zero_on_axis_0"),
        TileParams(reference_tests::Tensor(ET, {3}, std::vector<T>{1, 2, 3}),
                   reference_tests::Tensor(ET_INT, {3}, std::vector<T_INT>{2, 0, 3}),
                   reference_tests::Tensor(ET, {2, 0, 9}, std::vector<T>{}),
                   "tile_1d_to_3d_with_zero_on_axis_1"),
    };
    return params;
}

template <element::Type_t ET, element::Type_t ET_INT>
std::vector<TileParams> generateParamsFloatValue() {
    using T = typename element_type_traits<ET>::value_type;
    using T_INT = typename element_type_traits<ET_INT>::value_type;
    std::vector<TileParams> params{
        TileParams(reference_tests::Tensor(ET, {2, 1, 3}, std::vector<T>{1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f}),
                   reference_tests::Tensor(ET_INT, {2}, std::vector<T_INT>{2, 1}),
                   reference_tests::Tensor(
                       ET,
                       {2, 2, 3},
                       std::vector<T>{1.1f, 2.2f, 3.3f, 1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 4.4f, 5.5f, 6.6f}),
                   "tile_3d_to_3d_repeats_broadcast_float_val"),
    };
    return params;
}

std::vector<TileParams> generateCombinedParams() {
    const std::vector<std::vector<TileParams>> generatedParams{
        // test each data type for each repeats type
        generateParams<element::Type_t::i8, element::Type_t::i8>(),
        generateParams<element::Type_t::i8, element::Type_t::i16>(),
        generateParams<element::Type_t::i8, element::Type_t::i32>(),
        generateParams<element::Type_t::i8, element::Type_t::i64>(),
        generateParams<element::Type_t::i8, element::Type_t::u8>(),
        generateParams<element::Type_t::i8, element::Type_t::u16>(),
        generateParams<element::Type_t::i8, element::Type_t::u32>(),
        generateParams<element::Type_t::i8, element::Type_t::u64>(),
        generateParams<element::Type_t::i16, element::Type_t::i8>(),
        generateParams<element::Type_t::i16, element::Type_t::i16>(),
        generateParams<element::Type_t::i16, element::Type_t::i32>(),
        generateParams<element::Type_t::i16, element::Type_t::i64>(),
        generateParams<element::Type_t::i16, element::Type_t::u8>(),
        generateParams<element::Type_t::i16, element::Type_t::u16>(),
        generateParams<element::Type_t::i16, element::Type_t::u32>(),
        generateParams<element::Type_t::i16, element::Type_t::u64>(),
        generateParams<element::Type_t::i32, element::Type_t::i8>(),
        generateParams<element::Type_t::i32, element::Type_t::i16>(),
        generateParams<element::Type_t::i32, element::Type_t::i32>(),
        generateParams<element::Type_t::i32, element::Type_t::i64>(),
        generateParams<element::Type_t::i32, element::Type_t::u8>(),
        generateParams<element::Type_t::i32, element::Type_t::u16>(),
        generateParams<element::Type_t::i32, element::Type_t::u32>(),
        generateParams<element::Type_t::i32, element::Type_t::u64>(),
        generateParams<element::Type_t::i64, element::Type_t::i8>(),
        generateParams<element::Type_t::i64, element::Type_t::i16>(),
        generateParams<element::Type_t::i64, element::Type_t::i32>(),
        generateParams<element::Type_t::i64, element::Type_t::i64>(),
        generateParams<element::Type_t::i64, element::Type_t::u8>(),
        generateParams<element::Type_t::i64, element::Type_t::u16>(),
        generateParams<element::Type_t::i64, element::Type_t::u32>(),
        generateParams<element::Type_t::i64, element::Type_t::u64>(),
        generateParams<element::Type_t::u8, element::Type_t::i8>(),
        generateParams<element::Type_t::u8, element::Type_t::i16>(),
        generateParams<element::Type_t::u8, element::Type_t::i32>(),
        generateParams<element::Type_t::u8, element::Type_t::i64>(),
        generateParams<element::Type_t::u8, element::Type_t::u8>(),
        generateParams<element::Type_t::u8, element::Type_t::u16>(),
        generateParams<element::Type_t::u8, element::Type_t::u32>(),
        generateParams<element::Type_t::u8, element::Type_t::u64>(),
        generateParams<element::Type_t::u16, element::Type_t::i8>(),
        generateParams<element::Type_t::u16, element::Type_t::i16>(),
        generateParams<element::Type_t::u16, element::Type_t::i32>(),
        generateParams<element::Type_t::u16, element::Type_t::i64>(),
        generateParams<element::Type_t::u16, element::Type_t::u8>(),
        generateParams<element::Type_t::u16, element::Type_t::u16>(),
        generateParams<element::Type_t::u16, element::Type_t::u32>(),
        generateParams<element::Type_t::u16, element::Type_t::u64>(),
        generateParams<element::Type_t::u32, element::Type_t::i8>(),
        generateParams<element::Type_t::u32, element::Type_t::i16>(),
        generateParams<element::Type_t::u32, element::Type_t::i32>(),
        generateParams<element::Type_t::u32, element::Type_t::i64>(),
        generateParams<element::Type_t::u32, element::Type_t::u8>(),
        generateParams<element::Type_t::u32, element::Type_t::u16>(),
        generateParams<element::Type_t::u32, element::Type_t::u32>(),
        generateParams<element::Type_t::u32, element::Type_t::u64>(),
        generateParams<element::Type_t::u64, element::Type_t::i8>(),
        generateParams<element::Type_t::u64, element::Type_t::i16>(),
        generateParams<element::Type_t::u64, element::Type_t::i32>(),
        generateParams<element::Type_t::u64, element::Type_t::i64>(),
        generateParams<element::Type_t::u64, element::Type_t::u8>(),
        generateParams<element::Type_t::u64, element::Type_t::u16>(),
        generateParams<element::Type_t::u64, element::Type_t::u32>(),
        generateParams<element::Type_t::u64, element::Type_t::u64>(),
        generateParams<element::Type_t::f16, element::Type_t::i8>(),
        generateParams<element::Type_t::f16, element::Type_t::i16>(),
        generateParams<element::Type_t::f16, element::Type_t::i32>(),
        generateParams<element::Type_t::f16, element::Type_t::i64>(),
        generateParams<element::Type_t::f16, element::Type_t::u8>(),
        generateParams<element::Type_t::f16, element::Type_t::u16>(),
        generateParams<element::Type_t::f16, element::Type_t::u32>(),
        generateParams<element::Type_t::f16, element::Type_t::u64>(),
        generateParams<element::Type_t::f32, element::Type_t::i8>(),
        generateParams<element::Type_t::f32, element::Type_t::i16>(),
        generateParams<element::Type_t::f32, element::Type_t::i32>(),
        generateParams<element::Type_t::f32, element::Type_t::i64>(),
        generateParams<element::Type_t::f32, element::Type_t::u8>(),
        generateParams<element::Type_t::f32, element::Type_t::u16>(),
        generateParams<element::Type_t::f32, element::Type_t::u32>(),
        generateParams<element::Type_t::f32, element::Type_t::u64>(),
        generateParams<element::Type_t::f64, element::Type_t::i8>(),
        generateParams<element::Type_t::f64, element::Type_t::i16>(),
        generateParams<element::Type_t::f64, element::Type_t::i32>(),
        generateParams<element::Type_t::f64, element::Type_t::i64>(),
        generateParams<element::Type_t::f64, element::Type_t::u8>(),
        generateParams<element::Type_t::f64, element::Type_t::u16>(),
        generateParams<element::Type_t::f64, element::Type_t::u32>(),
        generateParams<element::Type_t::f64, element::Type_t::u64>(),
        generateParams<element::Type_t::bf16, element::Type_t::i8>(),
        generateParams<element::Type_t::bf16, element::Type_t::i16>(),
        generateParams<element::Type_t::bf16, element::Type_t::i32>(),
        generateParams<element::Type_t::bf16, element::Type_t::i64>(),
        generateParams<element::Type_t::bf16, element::Type_t::u8>(),
        generateParams<element::Type_t::bf16, element::Type_t::u16>(),
        generateParams<element::Type_t::bf16, element::Type_t::u32>(),
        generateParams<element::Type_t::bf16, element::Type_t::u64>(),
        // // test float values in data
        generateParamsFloatValue<element::Type_t::f16, element::Type_t::i8>(),
        generateParamsFloatValue<element::Type_t::f16, element::Type_t::i16>(),
        generateParamsFloatValue<element::Type_t::f16, element::Type_t::i32>(),
        generateParamsFloatValue<element::Type_t::f16, element::Type_t::i64>(),
        generateParamsFloatValue<element::Type_t::f16, element::Type_t::u8>(),
        generateParamsFloatValue<element::Type_t::f16, element::Type_t::u16>(),
        generateParamsFloatValue<element::Type_t::f16, element::Type_t::u32>(),
        generateParamsFloatValue<element::Type_t::f16, element::Type_t::u64>(),
        generateParamsFloatValue<element::Type_t::f32, element::Type_t::i8>(),
        generateParamsFloatValue<element::Type_t::f32, element::Type_t::i16>(),
        generateParamsFloatValue<element::Type_t::f32, element::Type_t::i32>(),
        generateParamsFloatValue<element::Type_t::f32, element::Type_t::i64>(),
        generateParamsFloatValue<element::Type_t::f32, element::Type_t::u8>(),
        generateParamsFloatValue<element::Type_t::f32, element::Type_t::u16>(),
        generateParamsFloatValue<element::Type_t::f32, element::Type_t::u32>(),
        generateParamsFloatValue<element::Type_t::f32, element::Type_t::u64>(),
        generateParamsFloatValue<element::Type_t::f64, element::Type_t::i8>(),
        generateParamsFloatValue<element::Type_t::f64, element::Type_t::i16>(),
        generateParamsFloatValue<element::Type_t::f64, element::Type_t::i32>(),
        generateParamsFloatValue<element::Type_t::f64, element::Type_t::i64>(),
        generateParamsFloatValue<element::Type_t::f64, element::Type_t::u8>(),
        generateParamsFloatValue<element::Type_t::f64, element::Type_t::u16>(),
        generateParamsFloatValue<element::Type_t::f64, element::Type_t::u32>(),
        generateParamsFloatValue<element::Type_t::f64, element::Type_t::u64>(),
        generateParamsFloatValue<element::Type_t::bf16, element::Type_t::i8>(),
        generateParamsFloatValue<element::Type_t::bf16, element::Type_t::i16>(),
        generateParamsFloatValue<element::Type_t::bf16, element::Type_t::i32>(),
        generateParamsFloatValue<element::Type_t::bf16, element::Type_t::i64>(),
        generateParamsFloatValue<element::Type_t::bf16, element::Type_t::u8>(),
        generateParamsFloatValue<element::Type_t::bf16, element::Type_t::u16>(),
        generateParamsFloatValue<element::Type_t::bf16, element::Type_t::u32>(),
        generateParamsFloatValue<element::Type_t::bf16, element::Type_t::u64>()};
    std::vector<TileParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Tile_With_Hardcoded_Refs,
                         ReferenceTileTest,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceTileTest::getTestCaseName);
}  // namespace
