// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/cum_sum.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct CumSumParams {
    // Custom axis input and attributes
    template <class IT, class AT>
    CumSumParams(const Shape& shape,
                 const element::Type& iType,
                 const std::vector<IT>& iValues,
                 const std::vector<IT>& oValues,
                 const bool execlusive,
                 const bool reverse,
                 const element::Type& axisType,
                 AT axisVal,
                 const Shape& axisShape)
        : execlusive(execlusive),
          reverse(reverse),
          axisValue(axisVal),
          axisShape(axisShape),
          inShape(shape),
          axisType(axisType),
          inType(iType),
          outType(iType),
          axisData(CreateTensor(axisType, std::vector<AT>{axisVal})),
          inputData(CreateTensor(shape, iType, iValues)),
          refData(CreateTensor(shape, iType, oValues)),
          testDefaults(false) {}

    // Default axis input and attributes
    template <class IT>
    CumSumParams(const Shape& shape,
                 const element::Type& iType,
                 const std::vector<IT>& iValues,
                 const std::vector<IT>& oValues)
        : inShape(shape),
          axisType(element::i32),
          inType(iType),
          outType(iType),
          inputData(CreateTensor(shape, iType, iValues)),
          refData(CreateTensor(shape, iType, oValues)),
          testDefaults(true) {}

    bool execlusive = false;
    bool reverse = false;
    int64_t axisValue = 0;

    Shape axisShape;
    Shape inShape;
    element::Type axisType;
    element::Type inType;
    element::Type outType;
    ov::Tensor axisData;
    ov::Tensor inputData;
    ov::Tensor refData;

    bool testDefaults = false;
};

class ReferenceCumSumLayerTest : public testing::TestWithParam<CumSumParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        legacy_compare = true;
        auto params = GetParam();
        if (params.testDefaults) {
            function = CreateFunction(params.inShape, params.inType);
            inputData = {params.inputData};
            refOutData = {params.refData};
        } else {
            function = CreateFunction(params.inShape,
                                      params.inType,
                                      params.axisShape,
                                      params.axisType,
                                      params.execlusive,
                                      params.reverse);
            inputData = {params.inputData, params.axisData};
            refOutData = {params.refData};
        }
    }
    static std::string getTestCaseName(const testing::TestParamInfo<CumSumParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "testDefaults=" << param.testDefaults << "_";
        result << "axisValue=" << param.axisValue << "_";
        result << "execlusive=" << param.execlusive << "_";
        result << "reverse=" << param.reverse << "_";
        result << "inShape=" << param.inShape << "_";
        result << "iType=" << param.inType << "_";
        result << "axisType=" << param.axisType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& data_shape,
                                                 const element::Type& data_type,
                                                 const Shape& axis_shape,
                                                 const element::Type& axis_type,
                                                 const bool execlusive,
                                                 const bool reverse) {
        const auto data_param = std::make_shared<op::v0::Parameter>(data_type, data_shape);
        const auto axis_param = std::make_shared<op::v0::Parameter>(axis_type, axis_shape);
        const auto cum_sum = std::make_shared<op::v0::CumSum>(data_param, axis_param, execlusive, reverse);
        return std::make_shared<ov::Model>(NodeVector{cum_sum}, ParameterVector{data_param, axis_param});
    }

    static std::shared_ptr<Model> CreateFunction(const Shape& data_shape, const element::Type& data_type) {
        const auto data_param = std::make_shared<op::v0::Parameter>(data_type, data_shape);
        const auto cum_sum = std::make_shared<op::v0::CumSum>(data_param);
        return std::make_shared<ov::Model>(NodeVector{cum_sum}, ParameterVector{data_param});
    }
};

TEST_P(ReferenceCumSumLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<CumSumParams> generateCumSumParams(const element::Type& type) {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<CumSumParams> opParams{
        // Default axis input and attributes
        CumSumParams(Shape{1}, type, std::vector<T>{3}, std::vector<T>{3}),
        CumSumParams(Shape{6}, type, std::vector<T>{1, 2, 3, 4, 5, 6}, std::vector<T>{1, 3, 6, 10, 15, 21}),
        CumSumParams(Shape{2, 4},
                     type,
                     std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7},
                     std::vector<T>{0, 1, 2, 3, 4, 6, 8, 10}),
        // Custom axis input and attributes
        CumSumParams(Shape{6},
                     type,
                     std::vector<T>{1, 2, 3, 4, 5, 6},
                     std::vector<T>{1, 3, 6, 10, 15, 21},
                     false,
                     false,
                     element::i32,
                     int32_t(0),
                     Shape{}),  // axis i32
        CumSumParams(Shape{6},
                     type,
                     std::vector<T>{1, 2, 3, 4, 5, 6},
                     std::vector<T>{1, 3, 6, 10, 15, 21},
                     false,
                     false,
                     element::i64,
                     int64_t(0),
                     Shape{}),  // axis i64
        CumSumParams(Shape{6},
                     type,
                     std::vector<T>{1, 2, 3, 4, 5, 6},
                     std::vector<T>{21, 20, 18, 15, 11, 6},
                     false,
                     true,
                     element::i64,
                     int64_t(0),
                     Shape{}),
        CumSumParams(Shape{6},
                     type,
                     std::vector<T>{1, 2, 3, 4, 5, 6},
                     std::vector<T>{0, 1, 3, 6, 10, 15},
                     true,
                     false,
                     element::i64,
                     int64_t(0),
                     Shape{}),
        CumSumParams(Shape{6},
                     type,
                     std::vector<T>{1, 2, 3, 4, 5, 6},
                     std::vector<T>{20, 18, 15, 11, 6, 0},
                     true,
                     true,
                     element::i64,
                     int64_t(0),
                     Shape{}),

        CumSumParams(Shape{2, 4},
                     type,
                     std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7},
                     std::vector<T>{0, 1, 2, 3, 4, 6, 8, 10},
                     false,
                     false,
                     element::i32,
                     int32_t(0),
                     Shape{}),
        CumSumParams(Shape{2, 4},
                     type,
                     std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7},
                     std::vector<T>{4, 6, 8, 10, 4, 5, 6, 7},
                     false,
                     true,
                     element::i32,
                     int32_t(0),
                     Shape{}),
        CumSumParams(Shape{2, 4},
                     type,
                     std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7},
                     std::vector<T>{0, 0, 0, 0, 0, 1, 2, 3},
                     true,
                     false,
                     element::i32,
                     int32_t(0),
                     Shape{}),
        CumSumParams(Shape{2, 4},
                     type,
                     std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7},
                     std::vector<T>{4, 5, 6, 7, 0, 0, 0, 0},
                     true,
                     true,
                     element::i32,
                     int32_t(0),
                     Shape{}),
        CumSumParams(Shape{2, 4},
                     type,
                     std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7},
                     std::vector<T>{0, 1, 3, 6, 4, 9, 15, 22},
                     false,
                     false,
                     element::i32,
                     int32_t(1),
                     Shape{}),
        CumSumParams(Shape{2, 4},
                     type,
                     std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7},
                     std::vector<T>{0, 0, 1, 3, 0, 4, 9, 15},
                     true,
                     false,
                     element::i32,
                     int32_t(1),
                     Shape{}),
        CumSumParams(Shape{2, 4},
                     type,
                     std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7},
                     std::vector<T>{6, 6, 5, 3, 22, 18, 13, 7},
                     false,
                     true,
                     element::i32,
                     int32_t(1),
                     Shape{}),
        CumSumParams(Shape{2, 4},
                     type,
                     std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7},
                     std::vector<T>{6, 5, 3, 0, 18, 13, 7, 0},
                     true,
                     true,
                     element::i32,
                     int32_t(1),
                     Shape{}),

        CumSumParams(
            Shape{3, 2, 4},
            type,
            std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
            std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 27, 30, 33, 36, 39, 42, 45},
            false,
            false,
            element::i32,
            int32_t(0),
            Shape{}),
        CumSumParams(
            Shape{3, 2, 4},
            type,
            std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
            std::vector<T>{0, 1, 2, 3, 4, 6, 8, 10, 8, 9, 10, 11, 20, 22, 24, 26, 16, 17, 18, 19, 36, 38, 40, 42},
            false,
            false,
            element::i32,
            int32_t(1),
            Shape{}),
        CumSumParams(
            Shape{3, 2, 4},
            type,
            std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
            std::vector<T>{0, 1, 3, 6, 4, 9, 15, 22, 8, 17, 27, 38, 12, 25, 39, 54, 16, 33, 51, 70, 20, 41, 63, 86},
            false,
            false,
            element::i32,
            int32_t(2),
            Shape{}),
    };
    return opParams;
}

std::vector<CumSumParams> generateCumSumCombinedParams() {
    const std::vector<std::vector<CumSumParams>> opTypeParams{
        generateCumSumParams<element::Type_t::bf16>(element::bf16),
        generateCumSumParams<element::Type_t::f16>(element::f16),
        generateCumSumParams<element::Type_t::f32>(element::f32),
        generateCumSumParams<element::Type_t::i32>(element::i32),
        generateCumSumParams<element::Type_t::i64>(element::i64),
        generateCumSumParams<element::Type_t::u32>(element::u32),
        generateCumSumParams<element::Type_t::i8>(element::i8)};
    std::vector<CumSumParams> combinedParams;
    std::for_each(opTypeParams.begin(), opTypeParams.end(), [&](std::vector<CumSumParams> params) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    });
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_CumSum_With_Hardcoded_Refs,
                         ReferenceCumSumLayerTest,
                         ::testing::ValuesIn(generateCumSumCombinedParams()),
                         ReferenceCumSumLayerTest::getTestCaseName);
}  // namespace
