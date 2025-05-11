// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/variadic_split.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct VariadicSplitParams {
    VariadicSplitParams(const ov::PartialShape& dynamicDataShape,
                        const reference_tests::Tensor& dataTensor,
                        const reference_tests::Tensor& axisTensor,
                        const reference_tests::Tensor& splitLengthTensor,
                        const std::vector<reference_tests::Tensor>& expectedTensors,
                        const std::string& testcaseName = "")
        : dynamicDataShape(dynamicDataShape),
          dataTensor(dataTensor),
          axisTensor(axisTensor),
          splitLengthTensor(splitLengthTensor),
          expectedTensors(expectedTensors),
          testcaseName(testcaseName) {}

    ov::PartialShape dynamicDataShape;
    reference_tests::Tensor dataTensor;
    reference_tests::Tensor axisTensor;
    reference_tests::Tensor splitLengthTensor;
    std::vector<reference_tests::Tensor> expectedTensors;
    std::string testcaseName;
};

class ReferenceVariadicSplitTest : public testing::TestWithParam<VariadicSplitParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        if (params.dynamicDataShape.is_static()) {
            inputData = {params.dataTensor.data};
            function = CreateFunction(params);
        } else {
            inputData = {params.dataTensor.data, params.axisTensor.data, params.splitLengthTensor.data};
            function = CreateDynamicFunction(params);
        }
        for (const auto& expectedTensor : params.expectedTensors) {
            refOutData.push_back(expectedTensor.data);
        }
    }

    static std::string getTestCaseName(const testing::TestParamInfo<VariadicSplitParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "ddShape=" << param.dynamicDataShape;
        result << "_dType=" << param.dataTensor.type;
        result << "_dShape=" << param.dataTensor.shape;
        result << "_aType=" << param.axisTensor.type;
        result << "_aShape=" << param.axisTensor.shape;
        result << "_sType=" << param.splitLengthTensor.type;
        result << "_sShape=" << param.splitLengthTensor.shape;
        result << "_eType=" << param.expectedTensors[0].type;
        if (param.testcaseName != "") {
            result << "_eShape=" << param.expectedTensors[0].shape;
            result << "_" << param.testcaseName;
        } else {
            result << "_eShape=" << param.expectedTensors[0].shape;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const VariadicSplitParams& params) {
        const auto data = std::make_shared<op::v0::Parameter>(params.dataTensor.type, params.dataTensor.shape);
        const auto axis = std::make_shared<op::v0::Constant>(params.axisTensor.type,
                                                             params.axisTensor.shape,
                                                             params.axisTensor.data.data());
        const auto splitLengths = std::make_shared<op::v0::Constant>(params.splitLengthTensor.type,
                                                                     params.splitLengthTensor.shape,
                                                                     params.splitLengthTensor.data.data());
        const auto split = std::make_shared<op::v1::VariadicSplit>(data, axis, splitLengths);
        return std::make_shared<ov::Model>(split->outputs(), ParameterVector{data});
    }

    static std::shared_ptr<Model> CreateDynamicFunction(const VariadicSplitParams& params) {
        const auto data = std::make_shared<op::v0::Parameter>(params.dataTensor.type, params.dynamicDataShape);
        const auto axis = std::make_shared<op::v0::Parameter>(params.axisTensor.type, params.axisTensor.shape);
        const auto splitLengths =
            std::make_shared<op::v0::Parameter>(params.splitLengthTensor.type, params.splitLengthTensor.shape);
        const auto split = std::make_shared<op::v1::VariadicSplit>(data, axis, splitLengths);
        return std::make_shared<ov::Model>(split->outputs(), ParameterVector{data, axis, splitLengths});
    }
};

TEST_P(ReferenceVariadicSplitTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<VariadicSplitParams> generateVariadicSplitParams() {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<VariadicSplitParams> variadicSplitParams{
        // variadic_split_1d_static
        VariadicSplitParams(
            {10},
            reference_tests::Tensor(IN_ET, {10}, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}),
            reference_tests::Tensor(element::i32, {1}, std::vector<int32_t>{0}),
            reference_tests::Tensor(element::i32, {3}, std::vector<int32_t>{5, 3, 2}),
            std::vector<reference_tests::Tensor>{reference_tests::Tensor(IN_ET, {5}, std::vector<T>{1, 2, 3, 4, 5}),
                                                 reference_tests::Tensor(IN_ET, {3}, std::vector<T>{6, 7, 8}),
                                                 reference_tests::Tensor(IN_ET, {2}, std::vector<T>{9, 10})},
            "variadic_split_1d_static"),
        // variadic_split_1d_dynamic
        VariadicSplitParams(
            PartialShape::dynamic(),
            reference_tests::Tensor(IN_ET, {10}, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}),
            reference_tests::Tensor(element::i32, {1}, std::vector<int32_t>{0}),
            reference_tests::Tensor(element::i32, {3}, std::vector<int32_t>{5, 3, 2}),
            std::vector<reference_tests::Tensor>{reference_tests::Tensor(IN_ET, {5}, std::vector<T>{1, 2, 3, 4, 5}),
                                                 reference_tests::Tensor(IN_ET, {3}, std::vector<T>{6, 7, 8}),
                                                 reference_tests::Tensor(IN_ET, {2}, std::vector<T>{9, 10})},
            "variadic_split_1d_dynamic"),
        // variadic_split_2d_axis_0_static
        VariadicSplitParams(
            {6, 2},
            reference_tests::Tensor(IN_ET, {6, 2}, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),
            reference_tests::Tensor(element::i32, {}, std::vector<int32_t>{0}),
            reference_tests::Tensor(element::i32, {2}, std::vector<int32_t>{4, 2}),
            std::vector<reference_tests::Tensor>{
                reference_tests::Tensor(IN_ET, {4, 2}, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8}),
                reference_tests::Tensor(IN_ET, {2, 2}, std::vector<T>{9, 10, 11, 12})},
            "variadic_split_2d_axis_0_static"),
        // variadic_split_2d_axis_0_dynamic
        VariadicSplitParams(
            PartialShape::dynamic(),
            reference_tests::Tensor(IN_ET, {6, 2}, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),
            reference_tests::Tensor(element::i32, {}, std::vector<int32_t>{0}),
            reference_tests::Tensor(element::i32, {2}, std::vector<int32_t>{4, 2}),
            std::vector<reference_tests::Tensor>{
                reference_tests::Tensor(IN_ET, {4, 2}, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8}),
                reference_tests::Tensor(IN_ET, {2, 2}, std::vector<T>{9, 10, 11, 12})},
            "variadic_split_2d_axis_0_dynamic"),
        // variadic_split_2d_axis_1_static
        VariadicSplitParams(
            {4, 3},
            reference_tests::Tensor(IN_ET, {4, 3}, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),
            reference_tests::Tensor(element::i32, {}, std::vector<int32_t>{1}),
            reference_tests::Tensor(element::i32, {2}, std::vector<int32_t>{1, 2}),
            std::vector<reference_tests::Tensor>{
                reference_tests::Tensor(IN_ET, {4, 1}, std::vector<T>{1, 4, 7, 10}),
                reference_tests::Tensor(IN_ET, {4, 2}, std::vector<T>{2, 3, 5, 6, 8, 9, 11, 12})},
            "variadic_split_2d_axis_1_static"),
        // variadic_split_2d_axis_1_dynamic
        VariadicSplitParams(
            PartialShape::dynamic(),
            reference_tests::Tensor(IN_ET, {4, 3}, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),
            reference_tests::Tensor(element::i32, {}, std::vector<int32_t>{1}),
            reference_tests::Tensor(element::i32, {2}, std::vector<int32_t>{1, 2}),
            std::vector<reference_tests::Tensor>{
                reference_tests::Tensor(IN_ET, {4, 1}, std::vector<T>{1, 4, 7, 10}),
                reference_tests::Tensor(IN_ET, {4, 2}, std::vector<T>{2, 3, 5, 6, 8, 9, 11, 12})},
            "variadic_split_2d_axis_1_dynamic"),
        // variadic_split_4d_axis_0_static
        VariadicSplitParams(
            {6, 2, 3, 1},
            reference_tests::Tensor(IN_ET, {6, 2, 3, 1}, std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,  8,
                                                                        9,  10, 11, 12, 13, 14, 15, 16, 17,
                                                                        18, 19, 20, 21, 22, 23, 24, 25, 26,
                                                                        27, 28, 29, 30, 31, 32, 33, 34, 35}),
            reference_tests::Tensor(element::i32, {1}, std::vector<int32_t>{0}),
            reference_tests::Tensor(element::i32, {3}, std::vector<int32_t>{3, 1, 2}),
            std::vector<reference_tests::Tensor>{
                reference_tests::Tensor(IN_ET,
                                        {3, 2, 3, 1},
                                        std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}),
                reference_tests::Tensor(IN_ET, {1, 2, 3, 1}, std::vector<T>{18, 19, 20, 21, 22, 23}),
                reference_tests::Tensor(IN_ET,
                                        {2, 2, 3, 1},
                                        std::vector<T>{24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35})},
            "variadic_split_4d_axis_0_static"),
        // variadic_split_4d_axis_0_dynamic
        VariadicSplitParams(
            PartialShape::dynamic(),
            reference_tests::Tensor(IN_ET, {6, 2, 3, 1}, std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,  8,
                                                                        9,  10, 11, 12, 13, 14, 15, 16, 17,
                                                                        18, 19, 20, 21, 22, 23, 24, 25, 26,
                                                                        27, 28, 29, 30, 31, 32, 33, 34, 35}),
            reference_tests::Tensor(element::i32, {1}, std::vector<int32_t>{0}),
            reference_tests::Tensor(element::i32, {3}, std::vector<int32_t>{3, 1, 2}),
            std::vector<reference_tests::Tensor>{
                reference_tests::Tensor(IN_ET,
                                        {3, 2, 3, 1},
                                        std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}),
                reference_tests::Tensor(IN_ET, {1, 2, 3, 1}, std::vector<T>{18, 19, 20, 21, 22, 23}),
                reference_tests::Tensor(IN_ET,
                                        {2, 2, 3, 1},
                                        std::vector<T>{24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35})},
            "variadic_split_4d_axis_0_dynamic"),
        // variadic_split_4d_axis_1_static
        VariadicSplitParams(
            {2, 8, 2, 2},
            reference_tests::Tensor(
                IN_ET,
                {2, 8, 2, 2},
                std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                               22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                               44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63}),
            reference_tests::Tensor(element::u64, {1}, std::vector<uint64_t>{1}),
            reference_tests::Tensor(element::u64, {4}, std::vector<uint64_t>{1, 3, 2, 2}),
            std::vector<reference_tests::Tensor>{
                reference_tests::Tensor(IN_ET, {2, 1, 2, 2}, std::vector<T>{0, 1, 2, 3, 32, 33, 34, 35}),
                reference_tests::Tensor(IN_ET, {2, 3, 2, 2}, std::vector<T>{4,  5,  6,  7,  8,  9,  10, 11,
                                                                            12, 13, 14, 15, 36, 37, 38, 39,
                                                                            40, 41, 42, 43, 44, 45, 46, 47}),
                reference_tests::Tensor(IN_ET,
                                        {2, 2, 2, 2},
                                        std::vector<T>{16, 17, 18, 19, 20, 21, 22, 23, 48, 49, 50, 51, 52, 53, 54, 55}),
                reference_tests::Tensor(
                    IN_ET,
                    {2, 2, 2, 2},
                    std::vector<T>{24, 25, 26, 27, 28, 29, 30, 31, 56, 57, 58, 59, 60, 61, 62, 63})},
            "variadic_split_4d_axis_1_static"),
        // variadic_split_4d_axis_1_dynamic
        VariadicSplitParams(
            PartialShape::dynamic(),
            reference_tests::Tensor(
                IN_ET,
                {2, 8, 2, 2},
                std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                               22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                               44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63}),
            reference_tests::Tensor(element::i64, {1}, std::vector<int64_t>{1}),
            reference_tests::Tensor(element::i64, {4}, std::vector<int64_t>{1, 3, 2, 2}),
            std::vector<reference_tests::Tensor>{
                reference_tests::Tensor(IN_ET, {2, 1, 2, 2}, std::vector<T>{0, 1, 2, 3, 32, 33, 34, 35}),
                reference_tests::Tensor(IN_ET, {2, 3, 2, 2}, std::vector<T>{4,  5,  6,  7,  8,  9,  10, 11,
                                                                            12, 13, 14, 15, 36, 37, 38, 39,
                                                                            40, 41, 42, 43, 44, 45, 46, 47}),
                reference_tests::Tensor(IN_ET,
                                        {2, 2, 2, 2},
                                        std::vector<T>{16, 17, 18, 19, 20, 21, 22, 23, 48, 49, 50, 51, 52, 53, 54, 55}),
                reference_tests::Tensor(
                    IN_ET,
                    {2, 2, 2, 2},
                    std::vector<T>{24, 25, 26, 27, 28, 29, 30, 31, 56, 57, 58, 59, 60, 61, 62, 63})},
            "variadic_split_4d_axis_1_dynamic"),
        // variadic_split_4d_axis_2_static
        VariadicSplitParams(
            {2, 1, 6, 2},
            reference_tests::Tensor(IN_ET, {2, 1, 6, 2}, std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,
                                                                        8,  9,  10, 11, 12, 13, 14, 15,
                                                                        16, 17, 18, 19, 20, 21, 22, 23}),
            reference_tests::Tensor(element::u32, {1}, std::vector<uint32_t>{2}),
            reference_tests::Tensor(element::u32, {3}, std::vector<uint32_t>{3, 1, 2}),
            std::vector<reference_tests::Tensor>{
                reference_tests::Tensor(IN_ET, {2, 1, 3, 2}, std::vector<T>{0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17}),
                reference_tests::Tensor(IN_ET, {2, 1, 1, 2}, std::vector<T>{6, 7, 18, 19}),
                reference_tests::Tensor(IN_ET, {2, 1, 2, 2}, std::vector<T>{8, 9, 10, 11, 20, 21, 22, 23})},
            "variadic_split_4d_axis_2_static"),
        // variadic_split_4d_axis_2_dynamic
        VariadicSplitParams(
            PartialShape::dynamic(),
            reference_tests::Tensor(IN_ET, {2, 1, 6, 2}, std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,
                                                                        8,  9,  10, 11, 12, 13, 14, 15,
                                                                        16, 17, 18, 19, 20, 21, 22, 23}),
            reference_tests::Tensor(element::i32, {1}, std::vector<int32_t>{2}),
            reference_tests::Tensor(element::i32, {3}, std::vector<int32_t>{-1, 1, 2}),
            std::vector<reference_tests::Tensor>{
                reference_tests::Tensor(IN_ET, {2, 1, 3, 2}, std::vector<T>{0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17}),
                reference_tests::Tensor(IN_ET, {2, 1, 1, 2}, std::vector<T>{6, 7, 18, 19}),
                reference_tests::Tensor(IN_ET, {2, 1, 2, 2}, std::vector<T>{8, 9, 10, 11, 20, 21, 22, 23})},
            "variadic_split_4d_axis_2_dynamic"),
        // variadic_split_4d_axis_3_static
        VariadicSplitParams(
            {2, 1, 2, 6},
            reference_tests::Tensor(IN_ET, {2, 1, 2, 6}, std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,
                                                                        8,  9,  10, 11, 12, 13, 14, 15,
                                                                        16, 17, 18, 19, 20, 21, 22, 23}),
            reference_tests::Tensor(element::i32, {1}, std::vector<int32_t>{3}),
            reference_tests::Tensor(element::i32, {3}, std::vector<int32_t>{1, -1, 3}),
            std::vector<reference_tests::Tensor>{
                reference_tests::Tensor(IN_ET, {2, 1, 2, 1}, std::vector<T>{0, 6, 12, 18}),
                reference_tests::Tensor(IN_ET, {2, 1, 2, 2}, std::vector<T>{1, 2, 7, 8, 13, 14, 19, 20}),
                reference_tests::Tensor(IN_ET,
                                        {2, 1, 2, 3},
                                        std::vector<T>{3, 4, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23})},
            "variadic_split_4d_axis_neg1_static"),
        // variadic_split_4d_axis_3_dynamic
        VariadicSplitParams(
            PartialShape::dynamic(),
            reference_tests::Tensor(IN_ET, {2, 1, 2, 6}, std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,
                                                                        8,  9,  10, 11, 12, 13, 14, 15,
                                                                        16, 17, 18, 19, 20, 21, 22, 23}),
            reference_tests::Tensor(element::i32, {1}, std::vector<int32_t>{-1}),
            reference_tests::Tensor(element::i32, {3}, std::vector<int32_t>{1, 2, -1}),
            std::vector<reference_tests::Tensor>{
                reference_tests::Tensor(IN_ET, {2, 1, 2, 1}, std::vector<T>{0, 6, 12, 18}),
                reference_tests::Tensor(IN_ET, {2, 1, 2, 2}, std::vector<T>{1, 2, 7, 8, 13, 14, 19, 20}),
                reference_tests::Tensor(IN_ET,
                                        {2, 1, 2, 3},
                                        std::vector<T>{3, 4, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23})},
            "variadic_split_4d_axis_3_dynamic"),
    };
    return variadicSplitParams;
}

std::vector<VariadicSplitParams> generateVariadicSplitCombinedParams() {
    const std::vector<std::vector<VariadicSplitParams>> variadicSplitTypeParams{
        generateVariadicSplitParams<element::Type_t::i8>(),
        generateVariadicSplitParams<element::Type_t::i16>(),
        generateVariadicSplitParams<element::Type_t::i32>(),
        generateVariadicSplitParams<element::Type_t::i64>(),
        generateVariadicSplitParams<element::Type_t::u8>(),
        generateVariadicSplitParams<element::Type_t::u16>(),
        generateVariadicSplitParams<element::Type_t::u32>(),
        generateVariadicSplitParams<element::Type_t::u64>(),
        generateVariadicSplitParams<element::Type_t::f16>(),
        generateVariadicSplitParams<element::Type_t::f32>(),
    };
    std::vector<VariadicSplitParams> combinedParams;

    for (const auto& params : variadicSplitTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit_With_Hardcoded_Refs,
                         ReferenceVariadicSplitTest,
                         testing::ValuesIn(generateVariadicSplitCombinedParams()),
                         ReferenceVariadicSplitTest::getTestCaseName);
}  // namespace
