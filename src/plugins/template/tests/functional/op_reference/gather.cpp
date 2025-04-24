// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gather.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct GatherParams {
    GatherParams(const reference_tests::Tensor& dataTensor,
                 const reference_tests::Tensor& indicesTensor,
                 const reference_tests::Tensor& axisTensor,
                 const reference_tests::Tensor& expectedTensor,
                 const std::string& testcaseName = "")
        : dataTensor(dataTensor),
          indicesTensor(indicesTensor),
          axisTensor(axisTensor),
          expectedTensor(expectedTensor),
          testcaseName(testcaseName) {}

    reference_tests::Tensor dataTensor;
    reference_tests::Tensor indicesTensor;
    reference_tests::Tensor axisTensor;
    reference_tests::Tensor expectedTensor;
    std::string testcaseName;
};

class ReferenceGatherTest : public testing::TestWithParam<GatherParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.dataTensor.data, params.indicesTensor.data};
        refOutData = {params.expectedTensor.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<GatherParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "dType=" << param.dataTensor.type;
        result << "_dShape=" << param.dataTensor.shape;
        result << "_iType=" << param.indicesTensor.type;
        result << "_iShape=" << param.indicesTensor.shape;
        result << "_aType=" << param.axisTensor.type;
        result << "_aShape=" << param.axisTensor.shape;
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
    static std::shared_ptr<Model> CreateFunction(const GatherParams& params) {
        const auto P = std::make_shared<op::v0::Parameter>(params.dataTensor.type, params.dataTensor.shape);
        const auto I = std::make_shared<op::v0::Parameter>(params.indicesTensor.type, params.indicesTensor.shape);
        const auto A =
            op::v0::Constant::create(params.axisTensor.type, params.axisTensor.shape, params.axisTensor.data.data());
        const auto G = std::make_shared<op::v1::Gather>(P, I, A);
        const auto f = std::make_shared<Model>(G, ParameterVector{P, I});
        return f;
    }
};

TEST_P(ReferenceGatherTest, CompareWithRefs) {
    Exec();
}

struct GatherParamsV7 {
    GatherParamsV7(const reference_tests::Tensor& dataTensor,
                   const reference_tests::Tensor& indicesTensor,
                   const reference_tests::Tensor& axisTensor,
                   int64_t batchDims,
                   const reference_tests::Tensor& expectedTensor,
                   const std::string& testcaseName = "")
        : dataTensor(dataTensor),
          indicesTensor(indicesTensor),
          axisTensor(axisTensor),
          batchDims(batchDims),
          expectedTensor(expectedTensor),
          testcaseName(testcaseName) {}

    reference_tests::Tensor dataTensor;
    reference_tests::Tensor indicesTensor;
    reference_tests::Tensor axisTensor;
    int64_t batchDims;
    reference_tests::Tensor expectedTensor;
    std::string testcaseName;
};

class ReferenceGatherTestV7 : public testing::TestWithParam<GatherParamsV7>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.dataTensor.data, params.indicesTensor.data};
        refOutData = {params.expectedTensor.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<GatherParamsV7>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "dType=" << param.dataTensor.type;
        result << "_dShape=" << param.dataTensor.shape;
        result << "_iType=" << param.indicesTensor.type;
        result << "_iShape=" << param.indicesTensor.shape;
        result << "_aType=" << param.axisTensor.type;
        result << "_aShape=" << param.axisTensor.shape;
        result << "_batchDims=" << param.batchDims;
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
    static std::shared_ptr<Model> CreateFunction(const GatherParamsV7& params) {
        const auto P = std::make_shared<op::v0::Parameter>(params.dataTensor.type, params.dataTensor.shape);
        const auto I = std::make_shared<op::v0::Parameter>(params.indicesTensor.type, params.indicesTensor.shape);
        const auto A =
            op::v0::Constant::create(params.axisTensor.type, params.axisTensor.shape, params.axisTensor.data.data());
        const auto G = std::make_shared<op::v7::Gather>(P, I, A, params.batchDims);
        const auto f = std::make_shared<Model>(G, ParameterVector{P, I});
        return f;
    }
};

TEST_P(ReferenceGatherTestV7, CompareWithRefs) {
    Exec();
}

class ReferenceGatherTestV8 : public ReferenceGatherTestV7 {
private:
    static std::shared_ptr<Model> CreateFunction(const GatherParamsV7& params) {
        const auto P = std::make_shared<op::v0::Parameter>(params.dataTensor.type, params.dataTensor.shape);
        const auto I = std::make_shared<op::v0::Parameter>(params.indicesTensor.type, params.indicesTensor.shape);
        const auto A =
            op::v0::Constant::create(params.axisTensor.type, params.axisTensor.shape, params.axisTensor.data.data());
        const auto G = std::make_shared<op::v8::Gather>(P, I, A, params.batchDims);
        const auto f = std::make_shared<Model>(G, ParameterVector{P, I});
        return f;
    }
};

TEST_P(ReferenceGatherTestV8, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET, element::Type_t ET_I, element::Type_t ET_A>
std::vector<GatherParams> generateParams() {
    using T = typename element_type_traits<ET>::value_type;
    using T_I = typename element_type_traits<ET_I>::value_type;
    using T_A = typename element_type_traits<ET_A>::value_type;
    std::vector<GatherParams> params{
        GatherParams(reference_tests::Tensor(ET, {3, 2}, std::vector<T>{10, 11, 20, 21, 30, 31}),
                     reference_tests::Tensor(ET_I, {2, 2, 3, 4}, std::vector<T_I>{0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2,
                                                                                  0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2,
                                                                                  0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2,
                                                                                  0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2}),
                     reference_tests::Tensor(ET_A, {}, std::vector<T_A>{0}),
                     reference_tests::Tensor(
                         ET,
                         {2, 2, 3, 4, 2},
                         std::vector<T>{10, 11, 20, 21, 20, 21, 30, 31, 10, 11, 20, 21, 20, 21, 30, 31, 10, 11, 20, 21,
                                        20, 21, 30, 31, 10, 11, 20, 21, 20, 21, 30, 31, 10, 11, 20, 21, 20, 21, 30, 31,
                                        10, 11, 20, 21, 20, 21, 30, 31, 10, 11, 20, 21, 20, 21, 30, 31, 10, 11, 20, 21,
                                        20, 21, 30, 31, 10, 11, 20, 21, 20, 21, 30, 31, 10, 11, 20, 21, 20, 21, 30, 31,
                                        10, 11, 20, 21, 20, 21, 30, 31, 10, 11, 20, 21, 20, 21, 30, 31}),
                     "gather_4d_indices_axis_0"),
        GatherParams(reference_tests::Tensor(ET, {3, 2}, std::vector<T>{10, 11, 20, 21, 30, 31}),
                     reference_tests::Tensor(ET_I, {2, 2}, std::vector<T_I>{0, 1, 1, 2}),
                     reference_tests::Tensor(ET_A, {}, std::vector<T_A>{0}),
                     reference_tests::Tensor(ET, {2, 2, 2}, std::vector<T>{10, 11, 20, 21, 20, 21, 30, 31}),
                     "gather_axis_0"),
    };
    return params;
}

template <element::Type_t ET, element::Type_t ET_I, element::Type_t ET_A>
std::vector<GatherParams> generateParamsFloatValue() {
    using T = typename element_type_traits<ET>::value_type;
    using T_I = typename element_type_traits<ET_I>::value_type;
    using T_A = typename element_type_traits<ET_A>::value_type;
    std::vector<GatherParams> params{
        GatherParams(reference_tests::Tensor(ET, {3, 2}, std::vector<T>{1.0f, 1.1f, 2.0f, 2.1f, 3.0f, 3.1f}),
                     reference_tests::Tensor(ET_I, {2, 2, 3, 4}, std::vector<T_I>{0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2,

                                                                                  0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2,

                                                                                  0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2,

                                                                                  0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2}),
                     reference_tests::Tensor(ET_A, {}, std::vector<T_A>{0}),
                     reference_tests::Tensor(ET,
                                             {2, 2, 3, 4, 2},
                                             std::vector<T>{1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,

                                                            1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,

                                                            1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,

                                                            1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,

                                                            1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,

                                                            1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,

                                                            1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,

                                                            1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,

                                                            1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,

                                                            1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,

                                                            1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,

                                                            1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f}),
                     "gather_4d_indices_axis_0_2d_input"),
        GatherParams(
            reference_tests::Tensor(ET, {3, 2}, std::vector<T>{1.0f, 1.1f, 2.0f, 2.1f, 3.0f, 3.1f}),
            reference_tests::Tensor(ET_I, {2, 3, 4}, std::vector<T_I>{0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2,

                                                                      0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2}),
            reference_tests::Tensor(ET_A, {}, std::vector<T_A>{0}),
            reference_tests::Tensor(ET, {2, 3, 4, 2}, std::vector<T>{1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,

                                                                     1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,

                                                                     1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,

                                                                     1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,

                                                                     1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,

                                                                     1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f}),
            "gather_3d_indices_axis_0_2d_input"),
        GatherParams(reference_tests::Tensor(ET, {3, 2}, std::vector<T>{1.0f, 1.1f, 2.0f, 2.1f, 3.0f, 3.1f}),
                     reference_tests::Tensor(ET_I, {2, 2}, std::vector<T_I>{0, 1, 1, 2}),
                     reference_tests::Tensor(ET_A, {}, std::vector<T_A>{0}),
                     reference_tests::Tensor(ET,
                                             {2, 2, 2},
                                             std::vector<T>{1.0f,
                                                            1.1f,
                                                            2.0f,
                                                            2.1f,

                                                            2.0f,
                                                            2.1f,
                                                            3.0f,
                                                            3.1f}),
                     "gather_2d_indices_axis_0_2d_input"),
        GatherParams(reference_tests::Tensor(ET, {3, 2}, std::vector<T>{1.0f, 1.1f, 2.0f, 2.1f, 3.0f, 3.1f}),
                     reference_tests::Tensor(ET_I, {2, 2}, std::vector<T_I>{0, -2, 1, 2}),
                     reference_tests::Tensor(ET_A, {}, std::vector<T_A>{0}),
                     reference_tests::Tensor(ET,
                                             {2, 2, 2},
                                             std::vector<T>{1.0f,
                                                            1.1f,
                                                            2.0f,
                                                            2.1f,

                                                            2.0f,
                                                            2.1f,
                                                            3.0f,
                                                            3.1f}),
                     "gather_2d_negative_and_positive_indices_axis_0_2d_input"),
        GatherParams(reference_tests::Tensor(ET, {3}, std::vector<T>{1.0f, 2.0f, 3.0f}),
                     reference_tests::Tensor(ET_I, {2}, std::vector<T_I>{1, 0}),
                     reference_tests::Tensor(ET_A, {}, std::vector<T_A>{0}),
                     reference_tests::Tensor(ET, {2}, std::vector<T>{2.0f, 1.0f}),
                     "gather_1d_indices_axis_0_1d_input"),
        GatherParams(reference_tests::Tensor(ET, {3, 2}, std::vector<T>{1.0f, 1.1f, 2.0f, 2.1f, 3.0f, 3.1f}),
                     reference_tests::Tensor(ET_I, {}, std::vector<T_I>{1}),
                     reference_tests::Tensor(ET_A, {}, std::vector<T_A>{0}),
                     reference_tests::Tensor(ET, {2}, std::vector<T>{2.0f, 2.1f}),
                     "gather_scalar_indices_axis_0_2d_input"),
        GatherParams(
            reference_tests::Tensor(ET, {3, 3}, std::vector<T>{1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f, 3.0f, 3.1f, 3.2f}),
            reference_tests::Tensor(ET_I, {1, 2}, std::vector<T_I>{0, 2}),
            reference_tests::Tensor(ET_A, {}, std::vector<T_A>{1}),
            reference_tests::Tensor(ET, {3, 1, 2}, std::vector<T>{1.0f, 1.2f, 2.0f, 2.2f, 3.0f, 3.2f}),
            "gather_2d_indices_axis_1_2d_input"),
        GatherParams(
            reference_tests::Tensor(
                ET,
                {2, 2, 3, 3},
                std::vector<T>{1.0f,   1.1f,   1.2f,   2.0f,   2.1f,   2.2f,   3.0f,   3.1f,   3.2f,

                               11.0f,  11.1f,  11.2f,  12.0f,  12.1f,  12.2f,  13.0f,  13.1f,  13.2f,

                               101.0f, 101.1f, 101.2f, 102.0f, 102.1f, 102.2f, 103.0f, 103.1f, 103.2f,

                               111.0f, 111.1f, 111.2f, 112.0f, 112.1f, 112.2f, 113.0f, 113.1f, 113.2f}),
            reference_tests::Tensor(ET_I, {2}, std::vector<T_I>{0, 2}),
            reference_tests::Tensor(ET_A, {}, std::vector<T_A>{2}),
            reference_tests::Tensor(ET, {2, 2, 2, 3}, std::vector<T>{1.0f,   1.1f,   1.2f,   3.0f,   3.1f,   3.2f,

                                                                     11.0f,  11.1f,  11.2f,  13.0f,  13.1f,  13.2f,

                                                                     101.0f, 101.1f, 101.2f, 103.0f, 103.1f, 103.2f,

                                                                     111.0f, 111.1f, 111.2f, 113.0f, 113.1f, 113.2f}),
            "gather_1d_indices_axis_2_4d_input"),
        GatherParams(
            reference_tests::Tensor(ET, {3, 3}, std::vector<T>{1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f, 3.0f, 3.1f, 3.2f}),
            reference_tests::Tensor(ET_I, {}, std::vector<T_I>{0}),
            reference_tests::Tensor(ET_A, {}, std::vector<T_A>{1}),
            reference_tests::Tensor(ET, {3}, std::vector<T>{1.0f, 2.0f, 3.0f}),
            "gather_scalar_indices_axis_1_2d_input"),
    };
    return params;
}

std::vector<GatherParams> generateCombinedParams() {
    const std::vector<std::vector<GatherParams>> generatedParams{
        generateParams<element::Type_t::i8, element::Type_t::i32, element::Type_t::i64>(),
        generateParams<element::Type_t::i16, element::Type_t::i32, element::Type_t::i64>(),
        generateParams<element::Type_t::i32, element::Type_t::i32, element::Type_t::i64>(),
        generateParams<element::Type_t::i64, element::Type_t::i32, element::Type_t::i64>(),
        generateParams<element::Type_t::u8, element::Type_t::i32, element::Type_t::i64>(),
        generateParams<element::Type_t::u16, element::Type_t::i32, element::Type_t::i64>(),
        generateParams<element::Type_t::u32, element::Type_t::i32, element::Type_t::i64>(),
        generateParams<element::Type_t::u64, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsFloatValue<element::Type_t::bf16, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsFloatValue<element::Type_t::f16, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsFloatValue<element::Type_t::f32, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsFloatValue<element::Type_t::f64, element::Type_t::i32, element::Type_t::i64>(),
    };
    std::vector<GatherParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Gather_With_Hardcoded_Refs,
                         ReferenceGatherTest,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceGatherTest::getTestCaseName);

template <element::Type_t ET, element::Type_t ET_I, element::Type_t ET_A>
std::vector<GatherParamsV7> generateParamsV7() {
    using T = typename element_type_traits<ET>::value_type;
    using T_I = typename element_type_traits<ET_I>::value_type;
    using T_A = typename element_type_traits<ET_A>::value_type;
    std::vector<GatherParamsV7> params{
        GatherParamsV7(reference_tests::Tensor(ET, {3}, std::vector<T>{1, 2, 3}),
                       reference_tests::Tensor(ET_I, {2}, std::vector<T_I>{2, 0}),
                       reference_tests::Tensor(ET_A, {}, std::vector<T_A>{0}),
                       0,
                       reference_tests::Tensor(ET, {2}, std::vector<T>{3, 1}),
                       "gather_v7_1d"),
        GatherParamsV7(reference_tests::Tensor(ET, {3, 2}, std::vector<T>{10, 11, 20, 21, 30, 31}),
                       reference_tests::Tensor(ET_I, {2, 2}, std::vector<T_I>{0, 1, 1, 2}),
                       reference_tests::Tensor(ET_A, {}, std::vector<T_A>{0}),
                       0,
                       reference_tests::Tensor(ET, {2, 2, 2}, std::vector<T>{10, 11, 20, 21, 20, 21, 30, 31}),
                       "gather_v7_axis_0"),
        GatherParamsV7(reference_tests::Tensor(ET, {2, 3}, std::vector<T>{1, 2, 3, 4, 5, 6}),
                       reference_tests::Tensor(ET_I,
                                               {2, 2, 2},
                                               std::vector<T_I>{0,
                                                                1,
                                                                1,
                                                                2,

                                                                2,
                                                                0,
                                                                1,
                                                                2}),
                       reference_tests::Tensor(ET_A, {}, std::vector<T_A>{1}),
                       1,
                       reference_tests::Tensor(ET,
                                               {2, 2, 2},
                                               std::vector<T>{1,
                                                              2,
                                                              2,
                                                              3,

                                                              6,
                                                              4,
                                                              5,
                                                              6}),
                       "gather_v7_data_int32_3d_indices_axis_1_batch_dims_1"),
        GatherParamsV7(reference_tests::Tensor(ET, {2, 5}, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}),
                       reference_tests::Tensor(ET_I, {2, 3}, std::vector<T_I>{0, 0, 4, 4, 0, 0}),
                       reference_tests::Tensor(ET_A, {}, std::vector<T_A>{1}),
                       1,
                       reference_tests::Tensor(ET, {2, 3}, std::vector<T>{1, 1, 5, 10, 6, 6}),
                       "gather_v7_data_int32_2d_indices_axis_1_batch_dims_1"),
        GatherParamsV7(reference_tests::Tensor(ET, {2, 5}, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}),
                       reference_tests::Tensor(ET_I, {2, 3}, std::vector<T_I>{0, 0, 4, 4, 0, 0}),
                       reference_tests::Tensor(ET_A, {}, std::vector<T_A>{1}),
                       -1,
                       reference_tests::Tensor(ET, {2, 3}, std::vector<T>{1, 1, 5, 10, 6, 6}),
                       "gather_v7_data_int32_2d_indices_axis_1_negative_batch_dims"),
        GatherParamsV7(
            reference_tests::Tensor(ET, {2, 1, 5, 4}, std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                                                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20,

                                                                     21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                                                     31, 32, 33, 34, 35, 36, 37, 38, 39, 40}),
            reference_tests::Tensor(ET_I, {2, 3}, std::vector<T_I>{1, 2, 4, 4, 3, 2}),
            reference_tests::Tensor(ET_A, {}, std::vector<T_A>{2}),
            1,
            reference_tests::Tensor(ET, {2, 1, 3, 4}, std::vector<T>{5,  6,  7,  8,  9,  10, 11, 12, 17, 18, 19, 20,

                                                                     37, 38, 39, 40, 33, 34, 35, 36, 29, 30, 31, 32}),
            "gather_v7_4d_data_axis_2_batch_dims_1_int32"),
    };
    return params;
}

template <>
std::vector<GatherParamsV7> generateParamsV7<element::boolean, element::i32, element::i64>() {
    std::vector<GatherParamsV7> params{
        GatherParamsV7(reference_tests::Tensor(element::boolean, {3, 2}, std::vector<char>{1, 1, 1, 0, 0, 1}),
                       reference_tests::Tensor(element::i32, {2, 2}, std::vector<int32_t>{0, 1, 1, 2}),
                       reference_tests::Tensor(element::i64, {}, std::vector<int64_t>{0}),
                       0,
                       reference_tests::Tensor(element::boolean, {2, 2, 2}, std::vector<char>{1, 1, 1, 0, 1, 0, 0, 1}),
                       "gather_v7_axis_0_bool"),
    };
    return params;
}

template <element::Type_t ET, element::Type_t ET_I, element::Type_t ET_A>
std::vector<GatherParamsV7> generateParamsStringValue() {
    using T = typename element_type_traits<ET>::value_type;
    using T_I = typename element_type_traits<ET_I>::value_type;
    using T_A = typename element_type_traits<ET_A>::value_type;
    std::vector<GatherParamsV7> params{
        {reference_tests::Tensor(ET, {2}, std::vector<T>{"A", "B c"}),
         reference_tests::Tensor(ET_I, {1}, std::vector<T_I>{1}),
         reference_tests::Tensor(ET_A, {1}, std::vector<T_A>{0}),
         0,
         reference_tests::Tensor(ET, {1}, std::vector<T>{"B c"}),
         "gather_string_1D_data"},
        {reference_tests::Tensor(ET, {2, 2}, std::vector<T>{"A", "B c", "d.Ef", " G h,i;"}),
         reference_tests::Tensor(ET_I, {1}, std::vector<T_I>{1}),
         reference_tests::Tensor(ET_A, {1}, std::vector<T_A>{0}),
         0,
         reference_tests::Tensor(ET, {1, 2}, std::vector<T>{"d.Ef", " G h,i;"}),
         "gather_string_2D_data"},
        {reference_tests::Tensor(ET, {2, 2}, std::vector<T>{"A", "B c", "d.Ef", " G h,i;"}),
         reference_tests::Tensor(ET_I, {2, 1}, std::vector<T_I>{0, 1}),
         reference_tests::Tensor(ET_A, {1}, std::vector<T_A>{1}),
         1,
         reference_tests::Tensor(ET, {2, 1}, std::vector<T>{"A", " G h,i;"}),
         "gather_string_2D_data_batch_dims_1"},
        {reference_tests::Tensor(ET, {2, 2}, std::vector<T>{"A", "B c", "d.Ef", " G h,i;"}),
         reference_tests::Tensor(ET_I, {2, 1}, std::vector<T_I>{1, 0}),
         reference_tests::Tensor(ET_A, {1}, std::vector<T_A>{1}),
         1,
         reference_tests::Tensor(ET, {2, 1}, std::vector<T>{"B c", "d.Ef"}),
         "gather_string_2D_data_batch_dims_1_reversed"},
        {reference_tests::Tensor(ET, {2, 1, 2}, std::vector<T>{"A", "B c", "d.Ef", " G h,i;"}),
         reference_tests::Tensor(ET_I, {2, 1, 2}, std::vector<T_I>{0, 1, 1, 0}),
         reference_tests::Tensor(ET_A, {1}, std::vector<T_A>{2}),
         2,
         reference_tests::Tensor(ET, {2, 1, 2}, std::vector<T>{"A", "B c", " G h,i;", "d.Ef"}),
         "gather_string_3D_data_batch_dims_2"},
        {reference_tests::Tensor(ET,
                                 {2, 2, 2},
                                 std::vector<T>{"A", "B c", "d.Ef", " G h,i;", "JK ", "l,m,n,", " ", " \0"}),
         reference_tests::Tensor(ET_I, {1}, std::vector<T_I>{1}),
         reference_tests::Tensor(ET_A, {1}, std::vector<T_A>{1}),
         0,
         reference_tests::Tensor(ET, {2, 1, 2}, std::vector<T>{"d.Ef", " G h,i;", " ", " \0"}),
         "gather_string_3D_data_axis_1"},
        {reference_tests::Tensor(ET,
                                 {2, 2, 2},
                                 std::vector<T>{"A", "B c", "d.Ef", " G h,i;", "JK ", "l,m,n,", " ", " \0"}),
         reference_tests::Tensor(ET_I, {1}, std::vector<T_I>{1}),
         reference_tests::Tensor(ET_A, {1}, std::vector<T_A>{0}),
         0,
         reference_tests::Tensor(ET, {1, 2, 2}, std::vector<T>{"JK ", "l,m,n,", " ", " \0"}),
         "gather_string_3D_data_axis_0"},
    };
    return params;
}

template <element::Type_t ET, element::Type_t ET_I, element::Type_t ET_A>
std::vector<GatherParamsV7> generateParamsFloatValueV7() {
    using T = typename element_type_traits<ET>::value_type;
    using T_I = typename element_type_traits<ET_I>::value_type;
    using T_A = typename element_type_traits<ET_A>::value_type;
    std::vector<GatherParamsV7> params{
        GatherParamsV7(
            reference_tests::Tensor(ET, {3, 2}, std::vector<T>{1.0f, 1.1f, 2.0f, 2.1f, 3.0f, 3.1f}),
            reference_tests::Tensor(ET_I, {2, 2, 3, 4}, std::vector<T_I>{0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2,

                                                                         0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2,

                                                                         0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2,

                                                                         0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2}),
            reference_tests::Tensor(ET_A, {}, std::vector<T_A>{0}),
            0,
            reference_tests::Tensor(ET,
                                    {2, 2, 3, 4, 2},
                                    std::vector<T>{1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,

                                                   1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,

                                                   1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,

                                                   1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,

                                                   1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,

                                                   1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,

                                                   1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,

                                                   1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,

                                                   1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,

                                                   1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,

                                                   1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,

                                                   1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f}),
            "gather_v7_4d_indices_axis_0_2d_input"),
        GatherParamsV7(
            reference_tests::Tensor(ET, {3, 2}, std::vector<T>{1.0f, 1.1f, 2.0f, 2.1f, 3.0f, 3.1f}),
            reference_tests::Tensor(ET_I, {2, 3, 4}, std::vector<T_I>{0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2,

                                                                      0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2}),
            reference_tests::Tensor(ET_A, {}, std::vector<T_A>{0}),
            0,
            reference_tests::Tensor(ET, {2, 3, 4, 2}, std::vector<T>{1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,

                                                                     1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,

                                                                     1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,

                                                                     1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,

                                                                     1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,

                                                                     1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f}),
            "gather_v7_3d_indices_axis_0_2d_input"),
        GatherParamsV7(reference_tests::Tensor(ET, {3, 2}, std::vector<T>{1.0f, 1.1f, 2.0f, 2.1f, 3.0f, 3.1f}),
                       reference_tests::Tensor(ET_I, {2, 2}, std::vector<T_I>{0, 1, 1, 2}),
                       reference_tests::Tensor(ET_A, {}, std::vector<T_A>{0}),
                       0,
                       reference_tests::Tensor(ET,
                                               {2, 2, 2},
                                               std::vector<T>{1.0f,
                                                              1.1f,
                                                              2.0f,
                                                              2.1f,

                                                              2.0f,
                                                              2.1f,
                                                              3.0f,
                                                              3.1f}),
                       "gather_v7_2d_indices_axis_0_2d_input"),
        GatherParamsV7(reference_tests::Tensor(ET, {3, 2}, std::vector<T>{1.0f, 1.1f, 2.0f, 2.1f, 3.0f, 3.1f}),
                       reference_tests::Tensor(ET_I, {2, 2}, std::vector<T_I>{0, -2, 1, 2}),
                       reference_tests::Tensor(ET_A, {}, std::vector<T_A>{0}),
                       0,
                       reference_tests::Tensor(ET,
                                               {2, 2, 2},
                                               std::vector<T>{1.0f,
                                                              1.1f,
                                                              2.0f,
                                                              2.1f,

                                                              2.0f,
                                                              2.1f,
                                                              3.0f,
                                                              3.1f}),
                       "gather_v7_2d_negative_and_positive_indices_axis_0_2d_input"),
        GatherParamsV7(reference_tests::Tensor(ET, {3}, std::vector<T>{1.0f, 2.0f, 3.0f}),
                       reference_tests::Tensor(ET_I, {2}, std::vector<T_I>{1, 0}),
                       reference_tests::Tensor(ET_A, {}, std::vector<T_A>{0}),
                       0,
                       reference_tests::Tensor(ET, {2}, std::vector<T>{2.0f, 1.0f}),
                       "gather_v7_1d_indices_axis_0_1d_input"),
        GatherParamsV7(reference_tests::Tensor(ET, {3, 2}, std::vector<T>{1.0f, 1.1f, 2.0f, 2.1f, 3.0f, 3.1f}),
                       reference_tests::Tensor(ET_I, {}, std::vector<T_I>{1}),
                       reference_tests::Tensor(ET_A, {}, std::vector<T_A>{0}),
                       0,
                       reference_tests::Tensor(ET, {2}, std::vector<T>{2.0f, 2.1f}),
                       "gather_v7_scalar_indices_axis_0_2d_input"),
        GatherParamsV7(
            reference_tests::Tensor(ET, {3, 3}, std::vector<T>{1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f, 3.0f, 3.1f, 3.2f}),
            reference_tests::Tensor(ET_I, {1, 2}, std::vector<T_I>{0, 2}),
            reference_tests::Tensor(ET_A, {}, std::vector<T_A>{1}),
            0,
            reference_tests::Tensor(ET, {3, 1, 2}, std::vector<T>{1.0f, 1.2f, 2.0f, 2.2f, 3.0f, 3.2f}),
            "gather_v7_2d_indices_axis_1_2d_input"),
        GatherParamsV7(
            reference_tests::Tensor(
                ET,
                {2, 2, 3, 3},
                std::vector<T>{1.0f,   1.1f,   1.2f,   2.0f,   2.1f,   2.2f,   3.0f,   3.1f,   3.2f,

                               11.0f,  11.1f,  11.2f,  12.0f,  12.1f,  12.2f,  13.0f,  13.1f,  13.2f,

                               101.0f, 101.1f, 101.2f, 102.0f, 102.1f, 102.2f, 103.0f, 103.1f, 103.2f,

                               111.0f, 111.1f, 111.2f, 112.0f, 112.1f, 112.2f, 113.0f, 113.1f, 113.2f}),
            reference_tests::Tensor(ET_I, {2}, std::vector<T_I>{0, 2}),
            reference_tests::Tensor(ET_A, {}, std::vector<T_A>{2}),
            0,
            reference_tests::Tensor(ET, {2, 2, 2, 3}, std::vector<T>{1.0f,   1.1f,   1.2f,   3.0f,   3.1f,   3.2f,

                                                                     11.0f,  11.1f,  11.2f,  13.0f,  13.1f,  13.2f,

                                                                     101.0f, 101.1f, 101.2f, 103.0f, 103.1f, 103.2f,

                                                                     111.0f, 111.1f, 111.2f, 113.0f, 113.1f, 113.2f}),
            "gather_v7_1d_indices_axis_2_4d_input"),
        GatherParamsV7(
            reference_tests::Tensor(ET, {3, 3}, std::vector<T>{1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f, 3.0f, 3.1f, 3.2f}),
            reference_tests::Tensor(ET_I, {}, std::vector<T_I>{0}),
            reference_tests::Tensor(ET_A, {}, std::vector<T_A>{1}),
            0,
            reference_tests::Tensor(ET, {3}, std::vector<T>{1.0f, 2.0f, 3.0f}),
            "gather_v7_scalar_indices_axis_1_2d_input"),
        GatherParamsV7(
            reference_tests::Tensor(ET, {2, 5, 2}, std::vector<T>{1.0f,  2.0f,  3.0f,  4.0f,  5.0f,
                                                                  6.0f,  7.0f,  8.0f,  9.0f,  10.0f,

                                                                  11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
                                                                  16.0f, 17.0f, 18.0f, 19.0f, 20.0f}),
            reference_tests::Tensor(ET_I,
                                    {2, 2, 3},
                                    std::vector<T_I>{0,
                                                     0,
                                                     4,
                                                     4,
                                                     0,
                                                     0,

                                                     1,
                                                     2,
                                                     4,
                                                     4,
                                                     3,
                                                     2}),
            reference_tests::Tensor(ET_A, {}, std::vector<T_A>{1}),
            1,
            reference_tests::Tensor(ET, {2, 2, 3, 2}, std::vector<T>{1.0f,  2.0f,  1.0f,  2.0f,  9.0f,  10.0f,

                                                                     9.0f,  10.0f, 1.0f,  2.0f,  1.0f,  2.0f,

                                                                     13.0f, 14.0f, 15.0f, 16.0f, 19.0f, 20.0f,

                                                                     19.0f, 20.0f, 17.0f, 18.0f, 15.0f, 16.0f}),
            "gather_v7_3d_indices_axis_1_batch_dims_1"),
    };
    return params;
}

std::vector<GatherParamsV7> generateCombinedParamsV7() {
    const std::vector<std::vector<GatherParamsV7>> generatedParams{
        generateParamsV7<element::Type_t::boolean, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsV7<element::Type_t::i8, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsV7<element::Type_t::i16, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsV7<element::Type_t::i32, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsV7<element::Type_t::i64, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsV7<element::Type_t::u8, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsV7<element::Type_t::u16, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsV7<element::Type_t::u32, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsV7<element::Type_t::u64, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsFloatValueV7<element::Type_t::bf16, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsFloatValueV7<element::Type_t::f16, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsFloatValueV7<element::Type_t::f32, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsFloatValueV7<element::Type_t::f64, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsStringValue<element::Type_t::string, element::Type_t::i32, element::Type_t::i64>(),
    };
    std::vector<GatherParamsV7> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Gather_With_Hardcoded_Refs,
                         ReferenceGatherTestV7,
                         testing::ValuesIn(generateCombinedParamsV7()),
                         ReferenceGatherTestV7::getTestCaseName);

template <element::Type_t ET, element::Type_t ET_I, element::Type_t ET_A>
std::vector<GatherParamsV7> generateParamsV8() {
    using T = typename element_type_traits<ET>::value_type;
    using T_I = typename element_type_traits<ET_I>::value_type;
    using T_A = typename element_type_traits<ET_A>::value_type;
    std::vector<GatherParamsV7> params{
        GatherParamsV7(reference_tests::Tensor(ET, {5}, std::vector<T>{1, 2, 3, 4, 5}),
                       reference_tests::Tensor(ET_I, {3}, std::vector<T_I>{0, -2, -1}),
                       reference_tests::Tensor(ET_A, {}, std::vector<T_A>{0}),
                       0,
                       reference_tests::Tensor(ET, {3}, std::vector<T>{1, 4, 5}),
                       "gather_v8_1d_negative_indices"),
        // zeros are not present in the original data but appear because of the out of bound values
        GatherParamsV7(reference_tests::Tensor(ET, {5}, std::vector<T>{1, 2, 3, 4, 5}),
                       reference_tests::Tensor(ET_I, {3}, std::vector<T_I>{0, 20, 4}),
                       reference_tests::Tensor(ET_A, {}, std::vector<T_A>{0}),
                       0,
                       reference_tests::Tensor(ET, {3}, std::vector<T>{1, 0, 5}),
                       "gather_v8_out_of_bound_indices_1"),
        GatherParamsV7(
            reference_tests::Tensor(ET, {2, 1, 5, 4}, std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                                                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20,

                                                                     21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                                                     31, 32, 33, 34, 35, 36, 37, 38, 39, 40}),
            reference_tests::Tensor(ET_I, {2, 3}, std::vector<T_I>{1, 2, 200, 4, 200, 2}),
            reference_tests::Tensor(ET_A, {}, std::vector<T_A>{2}),
            1,
            reference_tests::Tensor(ET, {2, 1, 3, 4}, std::vector<T>{5,  6,  7,  8,  9, 10, 11, 12, 0,  0,  0,  0,

                                                                     37, 38, 39, 40, 0, 0,  0,  0,  29, 30, 31, 32}),
            "gather_v8_4d_data_axis_2_batch_dims_1_out_of_bound_indices_2"),
        GatherParamsV7(reference_tests::Tensor(ET, {5}, std::vector<T>{1, 2, 3, 4, 5}),
                       reference_tests::Tensor(ET_I, {3}, std::vector<T_I>{0, -200, 4}),
                       reference_tests::Tensor(ET_A, {}, std::vector<T_A>{0}),
                       0,
                       reference_tests::Tensor(ET, {3}, std::vector<T>{1, 0, 5}),
                       "gather_v8_out_of_bound_negative_index"),
    };
    return params;
}

std::vector<GatherParamsV7> generateCombinedParamsV8() {
    const std::vector<std::vector<GatherParamsV7>> generatedParams{
        generateParamsV7<element::Type_t::boolean, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsV7<element::Type_t::i8, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsV7<element::Type_t::i16, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsV7<element::Type_t::i32, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsV7<element::Type_t::i64, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsV7<element::Type_t::u8, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsV7<element::Type_t::u16, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsV7<element::Type_t::u32, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsV7<element::Type_t::u64, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsFloatValueV7<element::Type_t::bf16, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsFloatValueV7<element::Type_t::f16, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsFloatValueV7<element::Type_t::f32, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsFloatValueV7<element::Type_t::f64, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsV8<element::Type_t::boolean, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsV8<element::Type_t::i8, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsV8<element::Type_t::i16, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsV8<element::Type_t::i32, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsV8<element::Type_t::i64, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsV8<element::Type_t::u8, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsV8<element::Type_t::u16, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsV8<element::Type_t::u32, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsV8<element::Type_t::u64, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsV8<element::Type_t::bf16, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsV8<element::Type_t::f16, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsV8<element::Type_t::f32, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsV8<element::Type_t::f64, element::Type_t::i32, element::Type_t::i64>(),
        generateParamsStringValue<element::Type_t::string, element::Type_t::i32, element::Type_t::i64>(),
    };
    std::vector<GatherParamsV7> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Gather_With_Hardcoded_Refs,
                         ReferenceGatherTestV8,
                         testing::ValuesIn(generateCombinedParamsV8()),
                         ReferenceGatherTestV8::getTestCaseName);
}  // namespace
