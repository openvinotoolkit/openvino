// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gather_nd.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct GatherNDParams {
    GatherNDParams(const reference_tests::Tensor& dataTensor,
                   const reference_tests::Tensor& indicesTensor,
                   int64_t batchDims,
                   const reference_tests::Tensor& expectedTensor,
                   const std::string& testcaseName = "")
        : dataTensor(dataTensor),
          indicesTensor(indicesTensor),
          batchDims(static_cast<int32_t>(batchDims)),
          expectedTensor(expectedTensor),
          testcaseName(testcaseName) {}

    reference_tests::Tensor dataTensor;
    reference_tests::Tensor indicesTensor;
    int32_t batchDims;
    reference_tests::Tensor expectedTensor;
    std::string testcaseName;
};

class ReferenceGatherND5Test : public testing::TestWithParam<GatherNDParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.dataTensor.data, params.indicesTensor.data};
        refOutData = {params.expectedTensor.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<GatherNDParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "dType=" << param.dataTensor.type;
        result << "_dShape=" << param.dataTensor.shape;
        result << "_aType=" << param.indicesTensor.type;
        result << "_aShape=" << param.indicesTensor.shape;
        result << "_bDims=" << param.batchDims;
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
    static std::shared_ptr<Model> CreateFunction(const GatherNDParams& params) {
        std::shared_ptr<Model> function;
        const auto data =
            std::make_shared<op::v0::Parameter>(params.dataTensor.type, PartialShape{params.dataTensor.shape});
        const auto indices =
            std::make_shared<op::v0::Parameter>(params.indicesTensor.type, PartialShape{params.indicesTensor.shape});
        std::shared_ptr<op::v5::GatherND> gatherND;
        if (params.batchDims == 0) {
            gatherND = std::make_shared<op::v5::GatherND>(data, indices);
        } else {
            gatherND = std::make_shared<op::v5::GatherND>(data, indices, params.batchDims);
        }
        function = std::make_shared<ov::Model>(NodeVector{gatherND}, ParameterVector{data, indices});
        return function;
    }
};

TEST_P(ReferenceGatherND5Test, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<GatherNDParams> generateParams() {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<GatherNDParams> params{
        GatherNDParams(reference_tests::Tensor(IN_ET, {3, 3}, std::vector<T>{10, 11, 12, 13, 14, 15, 16, 17, 18}),
                       reference_tests::Tensor(element::i32, {2}, std::vector<int32_t>{1, 2}),
                       0,
                       reference_tests::Tensor(IN_ET, {}, std::vector<T>{15}),
                       "gather_nd_single_indices"),
        GatherNDParams(reference_tests::Tensor(IN_ET, {2, 2}, std::vector<T>{10, 11, 12, 13}),
                       reference_tests::Tensor(element::i32, {2, 2}, std::vector<int32_t>{0, 0, 1, 1}),
                       0,
                       reference_tests::Tensor(IN_ET, {2}, std::vector<T>{10, 13}),
                       "gather_nd_scalar_from_2d"),
        GatherNDParams(reference_tests::Tensor(IN_ET, {2, 2}, std::vector<T>{10, 11, 12, 13}),
                       reference_tests::Tensor(element::i32, {2, 1}, std::vector<int32_t>{1, 0}),
                       0,
                       reference_tests::Tensor(IN_ET, {2, 2}, std::vector<T>{12, 13, 10, 11}),
                       "gather_nd_1d_from_2d"),
        GatherNDParams(reference_tests::Tensor(IN_ET, {2, 2, 2}, std::vector<T>{10, 11, 12, 13, 20, 21, 22, 23}),
                       reference_tests::Tensor(element::i32, {2, 3}, std::vector<int32_t>{0, 0, 1, 1, 0, 1}),
                       0,
                       reference_tests::Tensor(IN_ET, {2}, std::vector<T>{11, 21}),
                       "gather_nd_scalar_from_3d"),
        GatherNDParams(reference_tests::Tensor(IN_ET, {2, 2, 2}, std::vector<T>{10, 11, 12, 13, 20, 21, 22, 23}),
                       reference_tests::Tensor(element::i32, {2, 2}, std::vector<int32_t>{0, 1, 1, 0}),
                       0,
                       reference_tests::Tensor(IN_ET, {2, 2}, std::vector<T>{12, 13, 20, 21}),
                       "gather_nd_1d_from_3d"),
        GatherNDParams(reference_tests::Tensor(IN_ET, {2, 2, 2}, std::vector<T>{10, 11, 12, 13, 20, 21, 22, 23}),
                       reference_tests::Tensor(element::i32, {1, 1}, std::vector<int32_t>{1}),
                       0,
                       reference_tests::Tensor(IN_ET, {1, 2, 2}, std::vector<T>{20, 21, 22, 23}),
                       "gather_nd_2d_from_3d"),
        GatherNDParams(reference_tests::Tensor(IN_ET, {2, 2}, std::vector<T>{10, 11, 12, 13}),
                       reference_tests::Tensor(element::i32, {2, 1, 2}, std::vector<int32_t>{0, 0, 0, 1}),
                       0,
                       reference_tests::Tensor(IN_ET, {2, 1}, std::vector<T>{10, 11}),
                       "gather_nd_batch_scalar_from_2d"),
        GatherNDParams(reference_tests::Tensor(IN_ET, {2, 2}, std::vector<T>{10, 11, 12, 13}),
                       reference_tests::Tensor(element::i32, {2, 1, 1}, std::vector<int32_t>{1, 0}),
                       0,
                       reference_tests::Tensor(IN_ET, {2, 1, 2}, std::vector<T>{12, 13, 10, 11}),
                       "gather_nd_batch_1d_from_2d"),
        GatherNDParams(
            reference_tests::Tensor(IN_ET, {2, 2, 2}, std::vector<T>{10, 11, 12, 13, 20, 21, 22, 23}),
            reference_tests::Tensor(element::i32, {2, 2, 3}, std::vector<int32_t>{0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0}),
            0,
            reference_tests::Tensor(IN_ET, {2, 2}, std::vector<T>{11, 21, 13, 22}),
            "gather_nd_batch_scalar_from_3d"),
        GatherNDParams(reference_tests::Tensor(IN_ET, {2, 2, 2}, std::vector<T>{10, 11, 12, 13, 20, 21, 22, 23}),
                       reference_tests::Tensor(element::i32, {2, 2, 2}, std::vector<int32_t>{0, 1, 1, 0, 0, 0, 1, 1}),
                       0,
                       reference_tests::Tensor(IN_ET, {2, 2, 2}, std::vector<T>{12, 13, 20, 21, 10, 11, 22, 23}),
                       "gather_nd_batch_1d_from_3d"),
        GatherNDParams(reference_tests::Tensor(IN_ET, {2, 2, 2}, std::vector<T>{10, 11, 12, 13, 20, 21, 22, 23}),
                       reference_tests::Tensor(element::i32, {2, 2, 2}, std::vector<int32_t>{0, -1, -1, 0, 0, 0, 1, 1}),
                       0,
                       reference_tests::Tensor(IN_ET, {2, 2, 2}, std::vector<T>{12, 13, 20, 21, 10, 11, 22, 23}),
                       "gather_nd_batch_1d_from_3d_negative"),
        GatherNDParams(reference_tests::Tensor(IN_ET, {2, 2, 2}, std::vector<T>{10, 11, 12, 13, 20, 21, 22, 23}),
                       reference_tests::Tensor(element::i32, {2, 1, 1}, std::vector<int32_t>{1, 0}),
                       0,
                       reference_tests::Tensor(IN_ET, {2, 1, 2, 2}, std::vector<T>{20, 21, 22, 23, 10, 11, 12, 13}),
                       "gather_nd_batch_2d_from_3d"),
        GatherNDParams(
            reference_tests::Tensor(IN_ET, {2, 3, 4}, std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                                                     13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}),
            reference_tests::Tensor(element::i32, {2, 1}, std::vector<int32_t>{1, 0}),
            1,
            reference_tests::Tensor(IN_ET, {2, 4}, std::vector<T>{5, 6, 7, 8, 13, 14, 15, 16}),
            "gather_nd_batch_dims1"),
        GatherNDParams(
            reference_tests::Tensor(IN_ET,
                                    {2, 3, 4, 2},
                                    std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                                   17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                                                   33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48}),
            reference_tests::Tensor(element::i32,
                                    {2, 3, 3, 2},
                                    std::vector<int32_t>{1, 0, 3, 1, 2, 1, 0, 1, 1, 1, 2, 0, 3, 0, 3, 1, 2, 1,
                                                         2, 0, 1, 1, 3, 1, 1, 1, 2, 0, 2, 0, 0, 0, 3, 1, 3, 1}),
            2,
            reference_tests::Tensor(
                IN_ET,
                {6, 3},
                std::vector<T>{3, 8, 6, 10, 12, 13, 23, 24, 22, 29, 28, 32, 36, 37, 37, 41, 48, 48}),
            "gather_nd_batch_dims2"),
        GatherNDParams(
            reference_tests::Tensor(IN_ET, {2, 3, 4}, std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                                                     13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}),
            reference_tests::Tensor(element::i32, {2, 3, 1, 1}, std::vector<int32_t>{1, 0, 2, 0, 2, 2}),
            2,
            reference_tests::Tensor(IN_ET, {6, 1}, std::vector<T>{2, 5, 11, 13, 19, 23}),
            "gather_nd_batch_dims2_lead_dims"),
    };
    return params;
}

std::vector<GatherNDParams> generateCombinedParams() {
    const std::vector<std::vector<GatherNDParams>> generatedParams{
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
    std::vector<GatherNDParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_GatherND_With_Hardcoded_Refs,
                         ReferenceGatherND5Test,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceGatherND5Test::getTestCaseName);

class ReferenceGatherND8Test : public testing::TestWithParam<GatherNDParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.dataTensor.data, params.indicesTensor.data};
        refOutData = {params.expectedTensor.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<GatherNDParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "dType=" << param.dataTensor.type;
        result << "_dShape=" << param.dataTensor.shape;
        result << "_aType=" << param.indicesTensor.type;
        result << "_aShape=" << param.indicesTensor.shape;
        result << "_bDims=" << param.batchDims;
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
    static std::shared_ptr<Model> CreateFunction(const GatherNDParams& params) {
        std::shared_ptr<Model> function;
        const auto data =
            std::make_shared<op::v0::Parameter>(params.dataTensor.type, PartialShape{params.dataTensor.shape});
        const auto indices =
            std::make_shared<op::v0::Parameter>(params.indicesTensor.type, PartialShape{params.indicesTensor.shape});
        std::shared_ptr<op::v8::GatherND> gatherND;
        if (params.batchDims == 0) {
            gatherND = std::make_shared<op::v8::GatherND>(data, indices);
        } else {
            gatherND = std::make_shared<op::v8::GatherND>(data, indices, params.batchDims);
        }
        function = std::make_shared<ov::Model>(NodeVector{gatherND}, ParameterVector{data, indices});
        return function;
    }
};

TEST_P(ReferenceGatherND8Test, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<GatherNDParams> generateParams_v8() {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<GatherNDParams> params{
        GatherNDParams(reference_tests::Tensor(IN_ET, {3, 3}, std::vector<T>{10, 11, 12, 13, 14, 15, 16, 17, 18}),
                       reference_tests::Tensor(element::i32, {2}, std::vector<int32_t>{1, 2}),
                       0,
                       reference_tests::Tensor(IN_ET, {}, std::vector<T>{15}),
                       "gather_nd_8_single_indices"),
        GatherNDParams(reference_tests::Tensor(IN_ET, {2, 2}, std::vector<T>{10, 11, 12, 13}),
                       reference_tests::Tensor(element::i32, {2, 2}, std::vector<int32_t>{0, 0, 1, 1}),
                       0,
                       reference_tests::Tensor(IN_ET, {2}, std::vector<T>{10, 13}),
                       "gather_nd_8_scalar_from_2d"),
        GatherNDParams(reference_tests::Tensor(IN_ET, {2, 2}, std::vector<T>{10, 11, 12, 13}),
                       reference_tests::Tensor(element::i32, {2, 1}, std::vector<int32_t>{1, 0}),
                       0,
                       reference_tests::Tensor(IN_ET, {2, 2}, std::vector<T>{12, 13, 10, 11}),
                       "gather_nd_8_1d_from_2d"),
        GatherNDParams(reference_tests::Tensor(IN_ET, {2, 2, 2}, std::vector<T>{10, 11, 12, 13, 20, 21, 22, 23}),
                       reference_tests::Tensor(element::i32, {2, 3}, std::vector<int32_t>{0, 0, 1, 1, 0, 1}),
                       0,
                       reference_tests::Tensor(IN_ET, {2}, std::vector<T>{11, 21}),
                       "gather_nd_8_scalar_from_3d"),
        GatherNDParams(reference_tests::Tensor(IN_ET, {2, 2, 2}, std::vector<T>{10, 11, 12, 13, 20, 21, 22, 23}),
                       reference_tests::Tensor(element::i32, {2, 2}, std::vector<int32_t>{0, 1, 1, 0}),
                       0,
                       reference_tests::Tensor(IN_ET, {2, 2}, std::vector<T>{12, 13, 20, 21}),
                       "gather_nd_8_1d_from_3d"),
        GatherNDParams(reference_tests::Tensor(IN_ET, {2, 2, 2}, std::vector<T>{10, 11, 12, 13, 20, 21, 22, 23}),
                       reference_tests::Tensor(element::i32, {1, 1}, std::vector<int32_t>{1}),
                       0,
                       reference_tests::Tensor(IN_ET, {1, 2, 2}, std::vector<T>{20, 21, 22, 23}),
                       "gather_nd_8_2d_from_3d"),
        GatherNDParams(reference_tests::Tensor(IN_ET, {2, 2}, std::vector<T>{10, 11, 12, 13}),
                       reference_tests::Tensor(element::i32, {2, 1, 2}, std::vector<int32_t>{0, 0, 0, 1}),
                       0,
                       reference_tests::Tensor(IN_ET, {2, 1}, std::vector<T>{10, 11}),
                       "gather_nd_8_batch_scalar_from_2d"),
        GatherNDParams(reference_tests::Tensor(IN_ET, {2, 2}, std::vector<T>{10, 11, 12, 13}),
                       reference_tests::Tensor(element::i32, {2, 1, 1}, std::vector<int32_t>{1, 0}),
                       0,
                       reference_tests::Tensor(IN_ET, {2, 1, 2}, std::vector<T>{12, 13, 10, 11}),
                       "gather_nd_8_batch_1d_from_2d"),
        GatherNDParams(
            reference_tests::Tensor(IN_ET, {2, 2, 2}, std::vector<T>{10, 11, 12, 13, 20, 21, 22, 23}),
            reference_tests::Tensor(element::i32, {2, 2, 3}, std::vector<int32_t>{0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0}),
            0,
            reference_tests::Tensor(IN_ET, {2, 2}, std::vector<T>{11, 21, 13, 22}),
            "gather_nd_8_batch_scalar_from_3d"),
        GatherNDParams(reference_tests::Tensor(IN_ET, {2, 2, 2}, std::vector<T>{10, 11, 12, 13, 20, 21, 22, 23}),
                       reference_tests::Tensor(element::i32, {2, 2, 2}, std::vector<int32_t>{0, 1, 1, 0, 0, 0, 1, 1}),
                       0,
                       reference_tests::Tensor(IN_ET, {2, 2, 2}, std::vector<T>{12, 13, 20, 21, 10, 11, 22, 23}),
                       "gather_nd_8_batch_1d_from_3d"),
        GatherNDParams(reference_tests::Tensor(IN_ET, {2, 2, 2}, std::vector<T>{10, 11, 12, 13, 20, 21, 22, 23}),
                       reference_tests::Tensor(element::i32, {2, 2, 2}, std::vector<int32_t>{0, -1, -1, 0, 0, 0, 1, 1}),
                       0,
                       reference_tests::Tensor(IN_ET, {2, 2, 2}, std::vector<T>{12, 13, 20, 21, 10, 11, 22, 23}),
                       "gather_nd_8_batch_1d_from_3d_negative"),
        GatherNDParams(reference_tests::Tensor(IN_ET, {2, 2, 2}, std::vector<T>{10, 11, 12, 13, 20, 21, 22, 23}),
                       reference_tests::Tensor(element::i32, {2, 1, 1}, std::vector<int32_t>{1, 0}),
                       0,
                       reference_tests::Tensor(IN_ET, {2, 1, 2, 2}, std::vector<T>{20, 21, 22, 23, 10, 11, 12, 13}),
                       "gather_nd_8_batch_2d_from_3d"),
        GatherNDParams(
            reference_tests::Tensor(IN_ET, {2, 3, 4}, std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                                                     13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}),
            reference_tests::Tensor(element::i32, {2, 1}, std::vector<int32_t>{1, 0}),
            1,
            reference_tests::Tensor(IN_ET, {2, 4}, std::vector<T>{5, 6, 7, 8, 13, 14, 15, 16}),
            "gather_nd_8_batch_dims1"),
        GatherNDParams(
            reference_tests::Tensor(IN_ET,
                                    {2, 3, 4, 2},
                                    std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                                   17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                                                   33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48}),
            reference_tests::Tensor(element::i32,
                                    {2, 3, 3, 2},
                                    std::vector<int32_t>{1, 0, 3, 1, 2, 1, 0, 1, 1, 1, 2, 0, 3, 0, 3, 1, 2, 1,
                                                         2, 0, 1, 1, 3, 1, 1, 1, 2, 0, 2, 0, 0, 0, 3, 1, 3, 1}),
            2,
            reference_tests::Tensor(
                IN_ET,
                {2, 3, 3},
                std::vector<T>{3, 8, 6, 10, 12, 13, 23, 24, 22, 29, 28, 32, 36, 37, 37, 41, 48, 48}),
            "gather_8_nd_batch_dims2"),
        GatherNDParams(
            reference_tests::Tensor(IN_ET, {2, 3, 4}, std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                                                     13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}),
            reference_tests::Tensor(element::i32, {2, 3, 1, 1}, std::vector<int32_t>{1, 0, 2, 0, 2, 2}),
            2,
            reference_tests::Tensor(IN_ET, {2, 3, 1}, std::vector<T>{2, 5, 11, 13, 19, 23}),
            "gather_8_nd_batch_dims2_lead_dims"),
        GatherNDParams(
            reference_tests::Tensor(
                IN_ET,
                {2, 3, 4, 5},
                std::vector<T>{1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,
                               19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,
                               37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,
                               55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,
                               73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
                               91,  92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108,
                               109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120}),
            reference_tests::Tensor(element::i32,
                                    {2, 3, 2, 1},
                                    std::vector<int32_t>{1, 0, 2, 0, 2, 0, 1, 0, 2, 0, 2, 0}),
            2,
            reference_tests::Tensor(
                IN_ET,
                {2, 3, 2, 5},
                std::vector<T>{6,  7,  8,  9,  10, 1,   2,   3,   4,   5,   31,  32,  33,  34,  35,
                               21, 22, 23, 24, 25, 51,  52,  53,  54,  55,  41,  42,  43,  44,  45,
                               66, 67, 68, 69, 70, 61,  62,  63,  64,  65,  91,  92,  93,  94,  95,
                               81, 82, 83, 84, 85, 111, 112, 113, 114, 115, 101, 102, 103, 104, 105}),
            "gather_8_nd_batch_dims2_non_scalar_slices"),
    };
    return params;
}

std::vector<GatherNDParams> generateCombinedParams_v8() {
    const std::vector<std::vector<GatherNDParams>> generatedParams{
        generateParams_v8<element::Type_t::i8>(),
        generateParams_v8<element::Type_t::i16>(),
        generateParams_v8<element::Type_t::i32>(),
        generateParams_v8<element::Type_t::i64>(),
        generateParams_v8<element::Type_t::u8>(),
        generateParams_v8<element::Type_t::u16>(),
        generateParams_v8<element::Type_t::u32>(),
        generateParams_v8<element::Type_t::u64>(),
        generateParams_v8<element::Type_t::bf16>(),
        generateParams_v8<element::Type_t::f16>(),
        generateParams_v8<element::Type_t::f32>(),
        generateParams_v8<element::Type_t::f64>(),
    };
    std::vector<GatherNDParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_GatherND_With_Hardcoded_Refs,
                         ReferenceGatherND8Test,
                         testing::ValuesIn(generateCombinedParams_v8()),
                         ReferenceGatherND8Test::getTestCaseName);
}  // namespace
