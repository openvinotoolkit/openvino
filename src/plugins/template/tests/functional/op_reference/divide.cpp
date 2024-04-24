// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/divide.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace ov;
using namespace reference_tests;

namespace {

struct DivideParams {
    template <class IT>
    DivideParams(const Shape& iShape1,
                 const Shape& iShape2,
                 const Shape& oShape,
                 const element::Type& iType,
                 const std::vector<IT>& iValues1,
                 const std::vector<IT>& iValues2,
                 const std::vector<IT>& oValues)
        : pshape1(iShape1),
          pshape2(iShape2),
          inType(iType),
          outType(iType),
          inputData1(CreateTensor(iShape1, iType, iValues1)),
          inputData2(CreateTensor(iShape2, iType, iValues2)),
          refData(CreateTensor(oShape, iType, oValues)) {}

    Shape pshape1;
    Shape pshape2;
    element::Type inType;
    element::Type outType;
    ov::Tensor inputData1;
    ov::Tensor inputData2;
    ov::Tensor refData;
};

struct DivideRoundingParams : public DivideParams {
    template <class IT>
    DivideRoundingParams(const Shape& iShape1,
                         const Shape& iShape2,
                         const Shape& oShape,
                         const element::Type& iType,
                         const std::vector<IT>& iValues1,
                         const std::vector<IT>& iValues2,
                         const std::vector<IT>& oValues,
                         const bool pythondiv)
        : DivideParams(iShape1, iShape2, oShape, iType, iValues1, iValues2, oValues),
          pythonDivision(pythondiv) {}

    bool pythonDivision;
};

class ReferenceDivideLayerTest : public testing::TestWithParam<DivideParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape1, params.pshape2, params.inType, params.outType);
        inputData = {params.inputData1, params.inputData2};
        refOutData = {params.refData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<DivideParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "iShape1=" << param.pshape1 << "_";
        result << "iShape2=" << param.pshape2 << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& input_shape1,
                                                 const Shape& input_shape2,
                                                 const element::Type& input_type,
                                                 const element::Type& expected_output_type) {
        const auto in1 = std::make_shared<op::v0::Parameter>(input_type, input_shape1);
        const auto in2 = std::make_shared<op::v0::Parameter>(input_type, input_shape2);
        const auto divide = std::make_shared<op::v1::Divide>(in1, in2);
        return std::make_shared<Model>(NodeVector{divide}, ParameterVector{in1, in2});
    }
};

class ReferenceDivideRoundingLayerTest : public testing::TestWithParam<DivideRoundingParams>,
                                         public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape1, params.pshape2, params.inType, params.outType, params.pythonDivision);
        inputData = {params.inputData1, params.inputData2};
        refOutData = {params.refData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<DivideRoundingParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "iShape1=" << param.pshape1 << "_";
        result << "iShape2=" << param.pshape2 << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& input_shape1,
                                                 const Shape& input_shape2,
                                                 const element::Type& input_type,
                                                 const element::Type& expected_output_type,
                                                 const bool pythondiv) {
        const auto in1 = std::make_shared<op::v0::Parameter>(input_type, input_shape1);
        const auto in2 = std::make_shared<op::v0::Parameter>(input_type, input_shape2);
        const auto divide = std::make_shared<op::v1::Divide>(in1, in2, pythondiv);
        return std::make_shared<Model>(NodeVector{divide}, ParameterVector{in1, in2});
    }
};

TEST_P(ReferenceDivideLayerTest, DivideWithHardcodedRefs) {
    Exec();
}

TEST_P(ReferenceDivideRoundingLayerTest, DivideWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<DivideParams> generateParamsForDivide() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<DivideParams> params{DivideParams(ov::Shape{2, 2},
                                                  ov::Shape{2, 2},
                                                  ov::Shape{2, 2},
                                                  IN_ET,
                                                  std::vector<T>{2, 4, 8, 16},
                                                  std::vector<T>{1, 2, 4, 8},
                                                  std::vector<T>{2, 2, 2, 2})};
    return params;
}

template <element::Type_t IN_ET>
std::vector<DivideParams> generateParamsForDivideFloat32() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<DivideParams> params{DivideParams(ov::Shape{1},
                                                  ov::Shape{1},
                                                  ov::Shape{1},
                                                  IN_ET,
                                                  std::vector<T>{18},
                                                  std::vector<T>{8},
                                                  std::vector<T>{2.25}),
                                     DivideParams(ov::Shape{2, 2},
                                                  ov::Shape{2, 2},
                                                  ov::Shape{2, 2},
                                                  IN_ET,
                                                  std::vector<T>{2, 4, 8, 16},
                                                  std::vector<T>{0, 0, 0, 0},
                                                  std::vector<T>{std::numeric_limits<float>::infinity(),
                                                                 std::numeric_limits<float>::infinity(),
                                                                 std::numeric_limits<float>::infinity(),
                                                                 std::numeric_limits<float>::infinity()})};
    return params;
}

template <element::Type_t IN_ET>
std::vector<DivideParams> generateParamsForDivideInt32() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<DivideParams> params{DivideParams(ov::Shape{2, 2},
                                                  ov::Shape{2, 2},
                                                  ov::Shape{2, 2},
                                                  IN_ET,
                                                  std::vector<T>{0x40000140, 0x40000001, 8, 16},
                                                  std::vector<T>{2, 5, 4, 8},
                                                  std::vector<T>{536871072, 214748365, 2, 2}),
                                     DivideParams(ov::Shape{1},
                                                  ov::Shape{1},
                                                  ov::Shape{1},
                                                  IN_ET,
                                                  std::vector<T>{18},
                                                  std::vector<T>{8},
                                                  std::vector<T>{2})};
    return params;
}

template <element::Type_t IN_ET>
std::vector<DivideParams> generateParamsForDivideBroadcast() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<DivideParams> params{
        DivideParams(ov::Shape{3, 2, 1},
                     ov::Shape{1, 6},
                     ov::Shape{3, 2, 6},
                     IN_ET,
                     std::vector<T>{12, 24, 36, 48, 60, 72},
                     std::vector<T>{1, 2, 3, 4, 6, 1},
                     std::vector<T>{12, 6,  4,  3,  2, 12, 24, 12, 8,  6,  4,  24, 36, 18, 12, 9,  6,  36,
                                    48, 24, 16, 12, 8, 48, 60, 30, 20, 15, 10, 60, 72, 36, 24, 18, 12, 72})};
    return params;
}

template <element::Type_t IN_ET>
std::vector<DivideParams> generateParamsForDividePythonRoundingInt32() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<DivideParams> params{DivideParams(ov::Shape{2, 2},
                                                  ov::Shape{2, 2},
                                                  ov::Shape{2, 2},
                                                  IN_ET,
                                                  std::vector<T>{-10, -10, 10, 10},
                                                  std::vector<T>{-3, 3, -3, 3},
                                                  std::vector<T>{3, -4, -4, 3})};
    return params;
}

template <element::Type_t IN_ET>
std::vector<DivideRoundingParams> generateParamsForDivideCppRoundingInt32() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<DivideRoundingParams> params{DivideRoundingParams(ov::Shape{2, 2},
                                                                  ov::Shape{2, 2},
                                                                  ov::Shape{2, 2},
                                                                  IN_ET,
                                                                  std::vector<T>{-10, -10, 10, 10},
                                                                  std::vector<T>{-3, 3, -3, 3},
                                                                  std::vector<T>{3, -3, -3, 3},
                                                                  false)};
    return params;
}

std::vector<DivideParams> generateCombinedParamsForDivide() {
    const std::vector<std::vector<DivideParams>> allTypeParams{generateParamsForDivide<element::Type_t::i8>(),
                                                               generateParamsForDivide<element::Type_t::u8>(),
                                                               generateParamsForDivide<element::Type_t::f32>(),
                                                               generateParamsForDivide<element::Type_t::f16>(),
                                                               generateParamsForDivide<element::Type_t::bf16>(),
                                                               generateParamsForDivide<element::Type_t::i64>(),
                                                               generateParamsForDivide<element::Type_t::i32>(),
                                                               generateParamsForDivide<element::Type_t::u64>(),
                                                               generateParamsForDivide<element::Type_t::u32>()};

    std::vector<DivideParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

std::vector<DivideParams> generateCombinedParamsForDivideFloat32() {
    const std::vector<std::vector<DivideParams>> allTypeParams{generateParamsForDivideFloat32<element::Type_t::f32>()};

    std::vector<DivideParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

std::vector<DivideParams> generateCombinedParamsForDivideInt32() {
    const std::vector<std::vector<DivideParams>> allTypeParams{generateParamsForDivideInt32<element::Type_t::i32>()};

    std::vector<DivideParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

std::vector<DivideParams> generateCombinedParamsForDivideBroadcast() {
    const std::vector<std::vector<DivideParams>> allTypeParams{
        generateParamsForDivideBroadcast<element::Type_t::f32>(),
        generateParamsForDivideBroadcast<element::Type_t::i32>()};

    std::vector<DivideParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

std::vector<DivideParams> generateCombinedParamsForDividePythonRoundingInt32() {
    const std::vector<std::vector<DivideParams>> allTypeParams{
        generateParamsForDividePythonRoundingInt32<element::Type_t::i32>()};

    std::vector<DivideParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

std::vector<DivideRoundingParams> generateCombinedParamsForDivideCppRoundingInt32() {
    const std::vector<std::vector<DivideRoundingParams>> allTypeParams{
        generateParamsForDivideCppRoundingInt32<element::Type_t::i32>()};

    std::vector<DivideRoundingParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Divide_With_Hardcoded_Refs,
                         ReferenceDivideLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForDivide()),
                         ReferenceDivideLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Divide_Float32_With_Hardcoded_Refs,
                         ReferenceDivideLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForDivideFloat32()),
                         ReferenceDivideLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Divide_Int32_With_Hardcoded_Refs,
                         ReferenceDivideLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForDivideInt32()),
                         ReferenceDivideLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Divide_Broadcast_With_Hardcoded_Refs,
                         ReferenceDivideLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForDivideBroadcast()),
                         ReferenceDivideLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Divide_Python_Rounding_Int32_With_Hardcoded_Refs,
                         ReferenceDivideLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForDividePythonRoundingInt32()),
                         ReferenceDivideLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Divide_Cpp_Rounding_Int32_With_Hardcoded_Refs,
                         ReferenceDivideRoundingLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForDivideCppRoundingInt32()),
                         ReferenceDivideRoundingLayerTest::getTestCaseName);

}  // namespace
