// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gather_elements.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct GatherElementsParams {
    GatherElementsParams(const reference_tests::Tensor& dataTensor,
                         const reference_tests::Tensor& indicesTensor,
                         int64_t axis,
                         const reference_tests::Tensor& expectedTensor,
                         const std::string& testcaseName = "")
        : dataTensor(dataTensor),
          indicesTensor(indicesTensor),
          axis(axis),
          expectedTensor(expectedTensor),
          testcaseName(testcaseName) {}

    reference_tests::Tensor dataTensor;
    reference_tests::Tensor indicesTensor;
    int64_t axis;
    reference_tests::Tensor expectedTensor;
    std::string testcaseName;
};

class ReferenceGatherElementsTest : public testing::TestWithParam<GatherElementsParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.dataTensor.data, params.indicesTensor.data};
        refOutData = {params.expectedTensor.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<GatherElementsParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "dType=" << param.dataTensor.type;
        result << "_dShape=" << param.dataTensor.shape;
        result << "_aType=" << param.indicesTensor.type;
        result << "_aShape=" << param.indicesTensor.shape;
        result << "_axis=" << param.axis;
        result << "_eType=" << param.expectedTensor.type;
        if (param.testcaseName != "") {
            result << "_eShape=" << param.expectedTensor.shape;
            result << "_=" << param.testcaseName;
        } else {
            result << "_eShape=" << param.expectedTensor.shape;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const GatherElementsParams& params) {
        std::shared_ptr<Model> function;
        const auto data =
            std::make_shared<op::v0::Parameter>(params.dataTensor.type, PartialShape{params.dataTensor.shape});
        const auto indices =
            std::make_shared<op::v0::Parameter>(params.indicesTensor.type, PartialShape{params.indicesTensor.shape});
        const auto gatherElement = std::make_shared<op::v6::GatherElements>(data, indices, params.axis);
        function = std::make_shared<ov::Model>(NodeVector{gatherElement}, ParameterVector{data, indices});
        return function;
    }
};

class ReferenceGatherElementsTestNegative : public ReferenceGatherElementsTest {};

TEST_P(ReferenceGatherElementsTest, CompareWithRefs) {
    Exec();
}

TEST_P(ReferenceGatherElementsTestNegative, CompareWithRefs) {
    try {
        Exec();
    } catch (const std::domain_error& error) {
        ASSERT_EQ(error.what(), std::string("indices values of GatherElement exceed data size"));
    } catch (...) {
        FAIL() << "Evaluate out ouf bound indices check failed";
    }
}

template <element::Type_t IN_ET>
std::vector<GatherElementsParams> generateParams() {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<GatherElementsParams> params{
        GatherElementsParams(reference_tests::Tensor(IN_ET, {3}, std::vector<T>{1, 2, 3}),
                             reference_tests::Tensor(element::i32, {7}, std::vector<int32_t>{1, 2, 0, 2, 0, 0, 2}),
                             0,
                             reference_tests::Tensor(IN_ET, {7}, std::vector<T>{2, 3, 1, 3, 1, 1, 3}),
                             "evaluate_1D_gather_elements_3_indices_int32"),
        GatherElementsParams(reference_tests::Tensor(IN_ET, {2, 2}, std::vector<T>{1, 2, 3, 4}),
                             reference_tests::Tensor(element::i32, {2, 2}, std::vector<int32_t>{0, 1, 0, 0}),
                             0,
                             reference_tests::Tensor(IN_ET, {2, 2}, std::vector<T>{1, 4, 1, 2}),
                             "evaluate_2D_gather_elements_2x2_indices_int32_axis_0"),
        GatherElementsParams(reference_tests::Tensor(IN_ET, {2, 2}, std::vector<T>{1, 2, 3, 4}),
                             reference_tests::Tensor(element::i32, {2, 2}, std::vector<int32_t>{0, 1, 0, 0}),
                             1,
                             reference_tests::Tensor(IN_ET, {2, 2}, std::vector<T>{1, 2, 3, 3}),
                             "evaluate_2D_gather_elements_2x2_indices_int32_axis_1"),
        GatherElementsParams(reference_tests::Tensor(IN_ET, {2, 2}, std::vector<T>{1, 2, 3, 4}),
                             reference_tests::Tensor(element::i32, {2, 2}, std::vector<int32_t>{0, 1, 0, 0}),
                             -1,
                             reference_tests::Tensor(IN_ET, {2, 2}, std::vector<T>{1, 2, 3, 3}),
                             "evaluate_2D_gather_elements_2x2_indices_int32_axis_minus_1"),
        GatherElementsParams(reference_tests::Tensor(IN_ET, {3, 3}, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9}),
                             reference_tests::Tensor(element::i32, {2, 3}, std::vector<int32_t>{1, 2, 0, 2, 0, 0}),
                             0,
                             reference_tests::Tensor(IN_ET, {2, 3}, std::vector<T>{4, 8, 3, 7, 2, 3}),
                             "evaluate_2D_gather_elements_2x3_indices_int32"),
        GatherElementsParams(
            reference_tests::Tensor(IN_ET, {3, 2, 2}, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),
            reference_tests::Tensor(element::i32, {3, 2, 2}, std::vector<int32_t>{1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1}),
            -1,
            reference_tests::Tensor(IN_ET, {3, 2, 2}, std::vector<T>{2, 1, 3, 4, 6, 6, 8, 7, 9, 9, 12, 12}),
            "evaluate_3D_gather_elements_3x2x2_indices_int32"),
        GatherElementsParams(
            reference_tests::Tensor(IN_ET, {3, 2, 2, 2}, std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,
                                                                        9,  10, 11, 12, 13, 14, 15, 16,
                                                                        17, 18, 19, 20, 21, 22, 23, 24}),
            reference_tests::Tensor(
                element::i32,
                {3, 2, 2, 4},
                std::vector<int32_t>{1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0,
                                     0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0}),
            -1,
            reference_tests::Tensor(IN_ET,
                                    {3, 2, 2, 4},
                                    std::vector<T>{2,  1,  1,  1,  3,  4,  4,  3,  6,  6,  6,  6,  8,  7,  7,  8,
                                                   9,  9,  9,  10, 12, 12, 12, 11, 13, 13, 13, 13, 16, 15, 16, 15,
                                                   18, 18, 18, 18, 20, 19, 20, 19, 22, 21, 21, 22, 23, 23, 23, 23}),
            "evaluate_4D_gather_elements_3x2x2x2_indices_int64"),
        GatherElementsParams(
            reference_tests::Tensor(IN_ET, {3, 2, 2}, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),
            reference_tests::Tensor(element::i32, {3, 2, 2}, std::vector<int32_t>{1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1}),
            -1,
            reference_tests::Tensor(IN_ET, {3, 2, 2}, std::vector<T>{2, 1, 3, 4, 6, 6, 8, 7, 9, 9, 12, 12}),
            "evaluate_3D_gather_elements_3x2x2_indices_int64"),
        GatherElementsParams(reference_tests::Tensor(IN_ET, {3, 3}, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9}),
                             reference_tests::Tensor(element::i32, {2, 3}, std::vector<int32_t>{1, 2, 0, 2, 0, 0}),
                             0,
                             reference_tests::Tensor(IN_ET, {2, 3}, std::vector<T>{4, 8, 3, 7, 2, 3}),
                             "evaluate_2D_gather_elements_2x3_data_float32"),
        GatherElementsParams(
            reference_tests::Tensor(IN_ET, {2, 2, 1}, std::vector<T>{5, 4, 1, 4}),
            reference_tests::Tensor(element::i32, {4, 2, 1}, std::vector<int32_t>{0, 0, 1, 1, 1, 1, 0, 1}),
            0,
            reference_tests::Tensor(IN_ET, {4, 2, 1}, std::vector<T>{5, 4, 1, 4, 1, 4, 5, 4}),
            "evaluate_2D_gather_elements_2x2x1_data_float32"),
    };
    return params;
}

template <>
std::vector<GatherElementsParams> generateParams<element::Type_t::boolean>() {
    std::vector<GatherElementsParams> params{
        GatherElementsParams(
            reference_tests::Tensor(element::boolean, {3, 2}, std::vector<char>{true, false, true, true, false, false}),
            reference_tests::Tensor(element::i32, {2, 2}, std::vector<int32_t>{0, 1, 0, 2}),
            0,
            reference_tests::Tensor(element::boolean, {2, 2}, std::vector<char>{true, true, true, false}),
            "evaluate_2D_gather_elements_3x2_data_bool"),
    };
    return params;
}

std::vector<GatherElementsParams> generateParamsNegative() {
    std::vector<GatherElementsParams> params{
        GatherElementsParams(reference_tests::Tensor(element::i32, {3}, std::vector<int32_t>{1, 2, 3}),
                             reference_tests::Tensor(element::i32, {7}, std::vector<int32_t>{1, 2, 0, 2, 0, 0, 8}),
                             0,
                             reference_tests::Tensor(element::i32, {7}, std::vector<int32_t>{2, 3, 1, 3, 1, 1, 3}),
                             "evaluate_1D_gather_elements_negative_test"),
        GatherElementsParams(
            reference_tests::Tensor(element::i32, {3, 3}, std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9}),
            reference_tests::Tensor(element::i32, {2, 3}, std::vector<int32_t>{1, 3, 0, 2, 0, 0}),
            0,
            reference_tests::Tensor(element::i32, {2, 3}, std::vector<int32_t>{4, 8, 3, 7, 2, 3}),
            "evaluate_2D_gather_elements_negative_test"),
    };
    return params;
}

std::vector<GatherElementsParams> generateCombinedParams() {
    const std::vector<std::vector<GatherElementsParams>> generatedParams{
        generateParams<element::Type_t::boolean>(),
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
    std::vector<GatherElementsParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_With_Hardcoded_Refs,
                         ReferenceGatherElementsTest,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceGatherElementsTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_With_Hardcoded_Refs,
                         ReferenceGatherElementsTestNegative,
                         testing::ValuesIn(generateParamsNegative()),
                         ReferenceGatherElementsTest::getTestCaseName);
}  // namespace
