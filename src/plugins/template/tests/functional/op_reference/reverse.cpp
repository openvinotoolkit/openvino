// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reverse.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct ReverseParams {
    ReverseParams(const reference_tests::Tensor& constantTensor,
                  const op::v1::Reverse::Mode reverseMode,
                  const reference_tests::Tensor& dataTensor,
                  const reference_tests::Tensor& expectedTensor,
                  const std::string& testcaseName = "")
        : constantTensor(constantTensor),
          reverseMode(reverseMode),
          dataTensor(dataTensor),
          expectedTensor(expectedTensor),
          testcaseName(testcaseName) {}

    reference_tests::Tensor constantTensor;
    op::v1::Reverse::Mode reverseMode;
    reference_tests::Tensor dataTensor;
    reference_tests::Tensor expectedTensor;
    std::string testcaseName;
};

class ReferenceReverseTest : public testing::TestWithParam<ReverseParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.dataTensor.data};
        refOutData = {params.expectedTensor.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ReverseParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "cType=" << param.constantTensor.type << "_";
        result << "cShape=" << param.constantTensor.shape << "_";
        result << "rMode=" << param.reverseMode << "_";
        result << "dType=" << param.dataTensor.type << "_";
        result << "dShape=" << param.dataTensor.shape << "_";
        result << "eType=" << param.expectedTensor.type << "_";
        if (param.testcaseName != "") {
            result << "eShape=" << param.expectedTensor.shape << "_";
            result << "eShape=" << param.testcaseName;
        } else {
            result << "eShape=" << param.expectedTensor.shape;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ReverseParams& params) {
        const auto data = std::make_shared<op::v0::Parameter>(params.dataTensor.type, params.dataTensor.shape);
        const auto constant = std::make_shared<op::v0::Constant>(params.constantTensor.type,
                                                                 params.constantTensor.shape,
                                                                 params.constantTensor.data.data());
        const auto reverse = std::make_shared<op::v1::Reverse>(data, constant, params.reverseMode);
        return std::make_shared<ov::Model>(NodeVector{reverse}, ParameterVector{data});
    }
};

class ReferenceReverseTestAxesRankIndexMode : public ReferenceReverseTest {};

class ReferenceReverseTestAxesElemsMaskMode : public ReferenceReverseTest {};

class ReferenceReverseTestAxesOutOfBounds : public ReferenceReverseTest {};

class ReferenceReverseTestAxesOutOfBounds4 : public ReferenceReverseTest {};

TEST_P(ReferenceReverseTest, CompareWithRefs) {
    Exec();
}

TEST_P(ReferenceReverseTestAxesRankIndexMode, CompareWithRefs) {
    const auto Data = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 2, 2});
    const auto Rev_Axes = std::make_shared<op::v0::Parameter>(element::i64, Shape{1, 1});  // correct: 1D
    EXPECT_THROW(const auto unused = std::make_shared<ov::Model>(
                     std::make_shared<op::v1::Reverse>(Data, Rev_Axes, op::v1::Reverse::Mode::INDEX),
                     ParameterVector{Data, Rev_Axes}),
                 ov::NodeValidationFailure);
}

TEST_P(ReferenceReverseTestAxesElemsMaskMode, CompareWithRefs) {
    const auto Data = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 2, 2});
    const auto Rev_Axes = std::make_shared<op::v0::Parameter>(element::boolean, Shape{2});  // correct: 3
    EXPECT_THROW(const auto unused = std::make_shared<op::v1::Reverse>(Data, Rev_Axes, op::v1::Reverse::Mode::MASK),
                 ov::NodeValidationFailure);
}

TEST_P(ReferenceReverseTestAxesOutOfBounds, CompareWithRefs) {
    const auto Data = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 2, 2});
    const auto Rev_Axes = op::v0::Constant::create(element::i64, Shape{2}, {1, 10});
    EXPECT_THROW(const auto unused = std::make_shared<op::v1::Reverse>(Data, Rev_Axes, op::v1::Reverse::Mode::INDEX),
                 ov::NodeValidationFailure);
}

TEST_P(ReferenceReverseTestAxesOutOfBounds4, CompareWithRefs) {
    const auto Data = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 2, 2});
    const auto Rev_Axes = op::v0::Constant::create(element::i64, Shape{4}, {0, 1, 2, 3});
    EXPECT_THROW(const auto unused = std::make_shared<op::v1::Reverse>(Data, Rev_Axes, op::v1::Reverse::Mode::INDEX),
                 ov::NodeValidationFailure);
}

template <element::Type_t IN_ET>
std::vector<ReverseParams> generateParams() {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<ReverseParams> params{
        // nothing_to_reverse
        ReverseParams(reference_tests::Tensor({0}, element::i64, std::vector<int64_t>{}),
                      op::v1::Reverse::Mode::INDEX,
                      reference_tests::Tensor({8}, IN_ET, std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7}),
                      reference_tests::Tensor({8}, IN_ET, std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7}),
                      "nothing_to_reverse"),
        // reverse_1d
        ReverseParams(reference_tests::Tensor({1}, element::i64, std::vector<int64_t>{0}),
                      op::v1::Reverse::Mode::INDEX,
                      reference_tests::Tensor({8}, IN_ET, std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7}),
                      reference_tests::Tensor({8}, IN_ET, std::vector<T>{7, 6, 5, 4, 3, 2, 1, 0}),
                      "reverse_1d"),
        // reverse_2d_0
        ReverseParams(reference_tests::Tensor({1}, element::i64, std::vector<int64_t>{0}),
                      op::v1::Reverse::Mode::INDEX,
                      reference_tests::Tensor({4, 3}, IN_ET, std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}),
                      reference_tests::Tensor({4, 3}, IN_ET, std::vector<T>{9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2}),
                      "reverse_2d_0"),
        // reverse_2d_1
        ReverseParams(reference_tests::Tensor({1}, element::i64, std::vector<int64_t>{1}),
                      op::v1::Reverse::Mode::INDEX,
                      reference_tests::Tensor({4, 3}, IN_ET, std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}),
                      reference_tests::Tensor({4, 3}, IN_ET, std::vector<T>{2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9}),
                      "reverse_2d_1"),
        // reverse_2d_1_mask
        ReverseParams(reference_tests::Tensor({2}, element::boolean, std::vector<char>{false, true}),
                      op::v1::Reverse::Mode::MASK,
                      reference_tests::Tensor({4, 3}, IN_ET, std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}),
                      reference_tests::Tensor({4, 3}, IN_ET, std::vector<T>{2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9}),
                      "reverse_2d_1_mask"),
        // reverse_2d_01
        ReverseParams(reference_tests::Tensor({2}, element::i64, std::vector<int64_t>{0, 1}),
                      op::v1::Reverse::Mode::INDEX,
                      reference_tests::Tensor({4, 3}, IN_ET, std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}),
                      reference_tests::Tensor({4, 3}, IN_ET, std::vector<T>{11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}),
                      "reverse_2d_01"),
        // reverse_2d_01_mask
        ReverseParams(reference_tests::Tensor({2}, element::boolean, std::vector<char>{true, true}),
                      op::v1::Reverse::Mode::MASK,
                      reference_tests::Tensor({4, 3}, IN_ET, std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}),
                      reference_tests::Tensor({4, 3}, IN_ET, std::vector<T>{11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}),
                      "reverse_2d_01_mask"),
        // reverse_3d_0
        ReverseParams(
            reference_tests::Tensor({1}, element::i64, std::vector<int64_t>{0}),
            op::v1::Reverse::Mode::INDEX,
            reference_tests::Tensor({2, 4, 3}, IN_ET, std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                                     12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}),
            reference_tests::Tensor({2, 4, 3}, IN_ET, std::vector<T>{12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                                                     0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11}),
            "reverse_3d_0"),
        // reverse_3d_1
        ReverseParams(
            reference_tests::Tensor({1}, element::i64, std::vector<int64_t>{1}),
            op::v1::Reverse::Mode::INDEX,
            reference_tests::Tensor({2, 4, 3}, IN_ET, std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                                     12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}),
            reference_tests::Tensor({2, 4, 3}, IN_ET, std::vector<T>{9,  10, 11, 6,  7,  8,  3,  4,  5,  0,  1,  2,
                                                                     21, 22, 23, 18, 19, 20, 15, 16, 17, 12, 13, 14}),
            "reverse_3d_1"),
        // reverse_3d_2
        ReverseParams(
            reference_tests::Tensor({1}, element::i64, std::vector<int64_t>{2}),
            op::v1::Reverse::Mode::INDEX,
            reference_tests::Tensor({2, 4, 3}, IN_ET, std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                                     12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}),
            reference_tests::Tensor({2, 4, 3}, IN_ET, std::vector<T>{2,  1,  0,  5,  4,  3,  8,  7,  6,  11, 10, 9,
                                                                     14, 13, 12, 17, 16, 15, 20, 19, 18, 23, 22, 21}),
            "reverse_3d_2"),
        // reverse_3d_01
        ReverseParams(
            reference_tests::Tensor({2}, element::i64, std::vector<int64_t>{0, 1}),
            op::v1::Reverse::Mode::INDEX,
            reference_tests::Tensor({2, 4, 3}, IN_ET, std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                                     12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}),
            reference_tests::Tensor({2, 4, 3}, IN_ET, std::vector<T>{21, 22, 23, 18, 19, 20, 15, 16, 17, 12, 13, 14,
                                                                     9,  10, 11, 6,  7,  8,  3,  4,  5,  0,  1,  2}),
            "reverse_3d_01"),
        // reverse_3d_02
        ReverseParams(
            reference_tests::Tensor({2}, element::i64, std::vector<int64_t>{0, 2}),
            op::v1::Reverse::Mode::INDEX,
            reference_tests::Tensor({2, 4, 3}, IN_ET, std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                                     12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}),
            reference_tests::Tensor({2, 4, 3}, IN_ET, std::vector<T>{14, 13, 12, 17, 16, 15, 20, 19, 18, 23, 22, 21,
                                                                     2,  1,  0,  5,  4,  3,  8,  7,  6,  11, 10, 9}),
            "reverse_3d_02"),
        // reverse_3d_12
        ReverseParams(
            reference_tests::Tensor({2}, element::i64, std::vector<int64_t>{1, 2}),
            op::v1::Reverse::Mode::INDEX,
            reference_tests::Tensor({2, 4, 3}, IN_ET, std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                                     12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}),
            reference_tests::Tensor({2, 4, 3}, IN_ET, std::vector<T>{11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1,  0,
                                                                     23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12}),
            "reverse_3d_12"),
        // reverse_3d_012
        ReverseParams(
            reference_tests::Tensor({3}, element::i64, std::vector<int64_t>{0, 1, 2}),
            op::v1::Reverse::Mode::INDEX,
            reference_tests::Tensor({2, 4, 3}, IN_ET, std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                                     12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}),
            reference_tests::Tensor({2, 4, 3}, IN_ET, std::vector<T>{23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12,
                                                                     11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1,  0}),
            "reverse_3d_012"),
    };
    return params;
}

std::vector<ReverseParams> generateCombinedParams() {
    const std::vector<std::vector<ReverseParams>> reverseTypeParams{
        generateParams<element::Type_t::i8>(),
        generateParams<element::Type_t::i16>(),
        generateParams<element::Type_t::i32>(),
        generateParams<element::Type_t::i64>(),
        generateParams<element::Type_t::u8>(),
        generateParams<element::Type_t::u16>(),
        generateParams<element::Type_t::u32>(),
        generateParams<element::Type_t::u64>(),
        generateParams<element::Type_t::f16>(),
        generateParams<element::Type_t::f32>(),
    };
    std::vector<ReverseParams> combinedParams;

    for (const auto& params : reverseTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

std::vector<ReverseParams> generateParamsAxesRankIndexMode() {
    std::vector<ReverseParams> params{
        // reverse_v1_incorrect_rev_axes_rank_index_mode
        ReverseParams(reference_tests::Tensor({1}, element::i64, std::vector<int64_t>{0}),
                      op::v1::Reverse::Mode::INDEX,
                      reference_tests::Tensor({1}, element::i64, std::vector<int64_t>{0}),
                      reference_tests::Tensor({1}, element::i64, std::vector<int64_t>{0}),
                      "reverse_v1_incorrect_rev_axes_rank_index_mode"),
    };
    return params;
}

std::vector<ReverseParams> generateParamsAxesElemsMaskMode() {
    std::vector<ReverseParams> params{
        // reverse_v1_incorrect_rev_axes_elems_mask_mode
        ReverseParams(reference_tests::Tensor({1}, element::i64, std::vector<int64_t>{0}),
                      op::v1::Reverse::Mode::INDEX,
                      reference_tests::Tensor({1}, element::i64, std::vector<int64_t>{0}),
                      reference_tests::Tensor({1}, element::i64, std::vector<int64_t>{0}),
                      "reverse_v1_incorrect_rev_axes_elems_mask_mode"),
    };
    return params;
}

std::vector<ReverseParams> generateParamsAxesOutOfBounds() {
    std::vector<ReverseParams> params{
        // reverse_v1_axes_out_of_bounds
        ReverseParams(reference_tests::Tensor({1}, element::i64, std::vector<int64_t>{0}),
                      op::v1::Reverse::Mode::INDEX,
                      reference_tests::Tensor({1}, element::i64, std::vector<int64_t>{0}),
                      reference_tests::Tensor({1}, element::i64, std::vector<int64_t>{0}),
                      "reverse_v1_axes_out_of_bounds"),
    };
    return params;
}

std::vector<ReverseParams> generateParamsAxesOutOfBounds4() {
    std::vector<ReverseParams> params{
        // reverse_v1_axes_out_of_bounds_4
        ReverseParams(reference_tests::Tensor({1}, element::i64, std::vector<int64_t>{0}),
                      op::v1::Reverse::Mode::INDEX,
                      reference_tests::Tensor({1}, element::i64, std::vector<int64_t>{0}),
                      reference_tests::Tensor({1}, element::i64, std::vector<int64_t>{0}),
                      "reverse_v1_axes_out_of_bounds_4"),
    };
    return params;
}

INSTANTIATE_TEST_SUITE_P(smoke_Reverse_With_Hardcoded_Refs,
                         ReferenceReverseTest,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceReverseTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Reverse_With_Hardcoded_Refs,
                         ReferenceReverseTestAxesRankIndexMode,
                         testing::ValuesIn(generateParamsAxesRankIndexMode()),
                         ReferenceReverseTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Reverse_With_Hardcoded_Refs,
                         ReferenceReverseTestAxesElemsMaskMode,
                         testing::ValuesIn(generateParamsAxesElemsMaskMode()),
                         ReferenceReverseTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Reverse_With_Hardcoded_Refs,
                         ReferenceReverseTestAxesOutOfBounds,
                         testing::ValuesIn(generateParamsAxesOutOfBounds()),
                         ReferenceReverseTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Reverse_With_Hardcoded_Refs,
                         ReferenceReverseTestAxesOutOfBounds4,
                         testing::ValuesIn(generateParamsAxesOutOfBounds4()),
                         ReferenceReverseTest::getTestCaseName);
}  // namespace
