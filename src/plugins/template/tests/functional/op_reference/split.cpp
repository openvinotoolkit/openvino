// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/split.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct SplitParams {
    SplitParams(const reference_tests::Tensor& dataTensor,
                const reference_tests::Tensor& axisTensor,
                const size_t numSplits,
                const std::vector<reference_tests::Tensor>& expectedTensors,
                const std::string& testcaseName = "")
        : dataTensor(dataTensor),
          axisTensor(axisTensor),
          numSplits(numSplits),
          expectedTensors(expectedTensors),
          testcaseName(testcaseName) {}

    reference_tests::Tensor dataTensor;
    reference_tests::Tensor axisTensor;
    size_t numSplits;
    std::vector<reference_tests::Tensor> expectedTensors;
    std::string testcaseName;
};

class ReferenceSplitTest : public testing::TestWithParam<SplitParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.dataTensor.data};
        refOutData.reserve(params.expectedTensors.size());
        for (const auto& expectedTensor : params.expectedTensors) {
            refOutData.push_back(expectedTensor.data);
        }
    }

    static std::string getTestCaseName(const testing::TestParamInfo<SplitParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "dType=" << param.dataTensor.type << "_";
        result << "dShape=" << param.dataTensor.shape << "_";
        result << "aType=" << param.axisTensor.type << "_";
        result << "aShape=" << param.axisTensor.shape << "_";
        result << "nSplit=" << param.numSplits << "_";
        result << "eType=" << param.expectedTensors[0].type << "_";
        result << "eShape=" << param.expectedTensors[0].shape << "_";
        result << "eShape=" << param.testcaseName;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const SplitParams& params) {
        const auto data = std::make_shared<op::v0::Parameter>(params.dataTensor.type, params.dataTensor.shape);
        const auto axis = std::make_shared<op::v0::Constant>(params.axisTensor.type,
                                                             params.axisTensor.shape,
                                                             params.axisTensor.data.data());
        const auto split = std::make_shared<op::v1::Split>(data, axis, params.numSplits);
        return std::make_shared<ov::Model>(split->outputs(), ParameterVector{data});
    }
};

TEST_P(ReferenceSplitTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<SplitParams> generateSplitParams() {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<SplitParams> splitParams{
        // split_1d
        SplitParams(reference_tests::Tensor({6}, IN_ET, std::vector<T>{1, 2, 3, 4, 5, 6}),
                    reference_tests::Tensor({}, element::i64, std::vector<int64_t>{0}),
                    3,
                    std::vector<reference_tests::Tensor>{reference_tests::Tensor({2}, IN_ET, std::vector<T>{1, 2}),
                                                         reference_tests::Tensor({2}, IN_ET, std::vector<T>{3, 4}),
                                                         reference_tests::Tensor({2}, IN_ET, std::vector<T>{5, 6})},
                    "split_1d"),
        // split_2d_axis_0
        SplitParams(reference_tests::Tensor({6, 2}, IN_ET, std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}),
                    reference_tests::Tensor({}, element::i64, std::vector<int64_t>{0}),
                    2,
                    std::vector<reference_tests::Tensor>{
                        reference_tests::Tensor({3, 2}, IN_ET, std::vector<T>{0, 1, 2, 3, 4, 5}),
                        reference_tests::Tensor({3, 2}, IN_ET, std::vector<T>{6, 7, 8, 9, 10, 11})},
                    "split_2d_axis_0"),
        // split_2d_axis_1
        SplitParams(reference_tests::Tensor({6, 2}, IN_ET, std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}),
                    reference_tests::Tensor({}, element::i64, std::vector<int64_t>{1}),
                    2,
                    std::vector<reference_tests::Tensor>{
                        reference_tests::Tensor({6, 1}, IN_ET, std::vector<T>{0, 2, 4, 6, 8, 10}),
                        reference_tests::Tensor({6, 1}, IN_ET, std::vector<T>{1, 3, 5, 7, 9, 11})},
                    "split_2d_axis_1"),
        // split_3d_axis_0
        SplitParams(reference_tests::Tensor({2, 2, 3}, IN_ET, std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}),
                    reference_tests::Tensor({}, element::i64, std::vector<int64_t>{0}),
                    2,
                    std::vector<reference_tests::Tensor>{
                        reference_tests::Tensor({1, 2, 3}, IN_ET, std::vector<T>{0, 1, 2, 3, 4, 5}),
                        reference_tests::Tensor({1, 2, 3}, IN_ET, std::vector<T>{6, 7, 8, 9, 10, 11})},
                    "split_3d_axis_0"),
        // split_3d_axis_1
        SplitParams(reference_tests::Tensor({2, 8, 2}, IN_ET, std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                                                             11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                                                             22, 23, 24, 25, 26, 27, 28, 29, 30, 31}),
                    reference_tests::Tensor({}, element::i64, std::vector<int64_t>{1}),
                    4,
                    std::vector<reference_tests::Tensor>{
                        reference_tests::Tensor({2, 2, 2}, IN_ET, std::vector<T>{0, 1, 2, 3, 16, 17, 18, 19}),
                        reference_tests::Tensor({2, 2, 2}, IN_ET, std::vector<T>{4, 5, 6, 7, 20, 21, 22, 23}),
                        reference_tests::Tensor({2, 2, 2}, IN_ET, std::vector<T>{8, 9, 10, 11, 24, 25, 26, 27}),
                        reference_tests::Tensor({2, 2, 2}, IN_ET, std::vector<T>{12, 13, 14, 15, 28, 29, 30, 31})},
                    "split_3d_axis_1"),
        // split_3d_axis_2
        SplitParams(reference_tests::Tensor({2, 1, 6}, IN_ET, std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}),
                    reference_tests::Tensor({}, element::i64, std::vector<int64_t>{2}),
                    2,
                    std::vector<reference_tests::Tensor>{
                        reference_tests::Tensor({2, 1, 3}, IN_ET, std::vector<T>{0, 1, 2, 6, 7, 8}),
                        reference_tests::Tensor({2, 1, 3}, IN_ET, std::vector<T>{3, 4, 5, 9, 10, 11})},
                    "split_3d_axis_2"),
        // split_4d_axis_0
        SplitParams(
            reference_tests::Tensor({3, 2, 3, 1},
                                    IN_ET,
                                    std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}),
            reference_tests::Tensor({}, element::i64, std::vector<int64_t>{0}),
            3,
            std::vector<reference_tests::Tensor>{
                reference_tests::Tensor({1, 2, 3, 1}, IN_ET, std::vector<T>{0, 1, 2, 3, 4, 5}),
                reference_tests::Tensor({1, 2, 3, 1}, IN_ET, std::vector<T>{6, 7, 8, 9, 10, 11}),
                reference_tests::Tensor({1, 2, 3, 1}, IN_ET, std::vector<T>{12, 13, 14, 15, 16, 17})},
            "split_4d_axis_0"),
        // split_4d_axis_1
        SplitParams(
            reference_tests::Tensor(
                {2, 8, 2, 2},
                IN_ET,
                std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                               22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                               44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63}),
            reference_tests::Tensor({}, element::i64, std::vector<int64_t>{1}),
            4,
            std::vector<reference_tests::Tensor>{
                reference_tests::Tensor({2, 2, 2, 2},
                                        IN_ET,
                                        std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 32, 33, 34, 35, 36, 37, 38, 39}),
                reference_tests::Tensor({2, 2, 2, 2},
                                        IN_ET,
                                        std::vector<T>{8, 9, 10, 11, 12, 13, 14, 15, 40, 41, 42, 43, 44, 45, 46, 47}),
                reference_tests::Tensor({2, 2, 2, 2},
                                        IN_ET,
                                        std::vector<T>{16, 17, 18, 19, 20, 21, 22, 23, 48, 49, 50, 51, 52, 53, 54, 55}),
                reference_tests::Tensor(
                    {2, 2, 2, 2},
                    IN_ET,
                    std::vector<T>{24, 25, 26, 27, 28, 29, 30, 31, 56, 57, 58, 59, 60, 61, 62, 63})},
            "split_4d_axis_1"),
        // split_4d_axis_2
        SplitParams(reference_tests::Tensor({2, 1, 6, 2}, IN_ET, std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,
                                                                                8,  9,  10, 11, 12, 13, 14, 15,
                                                                                16, 17, 18, 19, 20, 21, 22, 23}),
                    reference_tests::Tensor({}, element::i64, std::vector<int64_t>{2}),
                    3,
                    std::vector<reference_tests::Tensor>{
                        reference_tests::Tensor({2, 1, 2, 2}, IN_ET, std::vector<T>{0, 1, 2, 3, 12, 13, 14, 15}),
                        reference_tests::Tensor({2, 1, 2, 2}, IN_ET, std::vector<T>{4, 5, 6, 7, 16, 17, 18, 19}),
                        reference_tests::Tensor({2, 1, 2, 2}, IN_ET, std::vector<T>{8, 9, 10, 11, 20, 21, 22, 23})},
                    "split_4d_axis_2"),
        // split_4d_axis_3
        SplitParams(reference_tests::Tensor({2, 1, 2, 6}, IN_ET, std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,
                                                                                8,  9,  10, 11, 12, 13, 14, 15,
                                                                                16, 17, 18, 19, 20, 21, 22, 23}),
                    reference_tests::Tensor({}, element::i64, std::vector<int64_t>{3}),
                    3,
                    std::vector<reference_tests::Tensor>{
                        reference_tests::Tensor({2, 1, 2, 2}, IN_ET, std::vector<T>{0, 1, 6, 7, 12, 13, 18, 19}),
                        reference_tests::Tensor({2, 1, 2, 2}, IN_ET, std::vector<T>{2, 3, 8, 9, 14, 15, 20, 21}),
                        reference_tests::Tensor({2, 1, 2, 2}, IN_ET, std::vector<T>{4, 5, 10, 11, 16, 17, 22, 23})},
                    "split_4d_axis_3"),
        // split_4d_axis_negative_2
        SplitParams(reference_tests::Tensor({2, 1, 4, 1}, IN_ET, std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7}),
                    reference_tests::Tensor({}, element::i32, std::vector<int32_t>{-2}),
                    2,
                    std::vector<reference_tests::Tensor>{
                        reference_tests::Tensor({2, 1, 2, 1}, IN_ET, std::vector<T>{0, 1, 4, 5}),
                        reference_tests::Tensor({2, 1, 2, 1}, IN_ET, std::vector<T>{2, 3, 6, 7})},
                    "split_4d_axis_negative_2"),
    };
    return splitParams;
}

std::vector<SplitParams> generateSplitCombinedParams() {
    const std::vector<std::vector<SplitParams>> splitTypeParams{
        generateSplitParams<element::Type_t::boolean>(),
        generateSplitParams<element::Type_t::i8>(),
        generateSplitParams<element::Type_t::i16>(),
        generateSplitParams<element::Type_t::i32>(),
        generateSplitParams<element::Type_t::i64>(),
        generateSplitParams<element::Type_t::u8>(),
        generateSplitParams<element::Type_t::u16>(),
        generateSplitParams<element::Type_t::u32>(),
        generateSplitParams<element::Type_t::u64>(),
        generateSplitParams<element::Type_t::f16>(),
        generateSplitParams<element::Type_t::f32>(),
    };
    std::vector<SplitParams> combinedParams;

    for (const auto& params : splitTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Split_With_Hardcoded_Refs,
                         ReferenceSplitTest,
                         testing::ValuesIn(generateSplitCombinedParams()),
                         ReferenceSplitTest::getTestCaseName);
}  // namespace
