// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/floor_mod.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace ov;
using namespace reference_tests;

namespace {

struct FloorModParams {
    template <class IT>
    FloorModParams(const PartialShape& iShape1,
                   const PartialShape& iShape2,
                   const Shape& oShape,
                   const element::Type& iType,
                   const std::vector<IT>& iValues1,
                   const std::vector<IT>& iValues2,
                   const std::vector<IT>& oValues)
        : pshape1(iShape1),
          pshape2(iShape2),
          inType(iType),
          outType(iType),
          inputData1(CreateTensor(iType, iValues1)),
          inputData2(CreateTensor(iType, iValues2)),
          refData(CreateTensor(oShape, iType, oValues)) {}

    PartialShape pshape1;
    PartialShape pshape2;
    element::Type inType;
    element::Type outType;
    ov::Tensor inputData1;
    ov::Tensor inputData2;
    ov::Tensor refData;
};

class ReferenceFloorModLayerTest : public testing::TestWithParam<FloorModParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params.pshape1, params.pshape2, params.inType, params.outType);
        inputData = {params.inputData1, params.inputData2};
        refOutData = {params.refData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<FloorModParams>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << "iShape1=" << param.pshape1 << "_";
        result << "iShape2=" << param.pshape2 << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const PartialShape& input_shape1,
                                                 const PartialShape& input_shape2,
                                                 const element::Type& input_type,
                                                 const element::Type& expected_output_type) {
        const auto in1 = std::make_shared<op::v0::Parameter>(input_type, input_shape1);
        const auto in2 = std::make_shared<op::v0::Parameter>(input_type, input_shape2);
        const auto floormod = std::make_shared<op::v1::FloorMod>(in1, in2);

        return std::make_shared<Model>(NodeVector{floormod}, ParameterVector{in1, in2});
    }
};

TEST_P(ReferenceFloorModLayerTest, DivideWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<FloorModParams> generateParamsForFloorMod() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<FloorModParams> params{FloorModParams(ov::PartialShape{4},
                                                      ov::PartialShape{4},
                                                      ov::Shape{4},
                                                      IN_ET,
                                                      std::vector<T>{7, -7, 7, -7},
                                                      std::vector<T>{3, 3, -3, -3},
                                                      std::vector<T>{1, 2, -2, -1})};
    return params;
}

template <element::Type_t IN_ET>
std::vector<FloorModParams> generateParamsForFloorModBroadcast() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<FloorModParams> params{
        FloorModParams(ov::PartialShape{2, 1, 2},
                       ov::PartialShape{2, 1},
                       ov::Shape{2, 2, 2},
                       IN_ET,
                       std::vector<T>{1, 2, 3, 4},
                       std::vector<T>{2, 3},
                       std::vector<T>{1, 0, 1, 2, 1, 0, 0, 1}),
    };
    return params;
}

template <element::Type_t IN_ET>
std::vector<FloorModParams> generateParamsForFloorModScalar() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<FloorModParams> params{
        FloorModParams(ov::PartialShape{1},
                       ov::PartialShape{1},
                       ov::Shape{1},
                       IN_ET,
                       std::vector<T>{2},
                       std::vector<T>{4},
                       std::vector<T>{2}),
    };
    return params;
}

template <element::Type_t IN_ET>
std::vector<FloorModParams> generateParamsForFloorModNonIntegerDivisor() {
    using T = typename element_type_traits<IN_ET>::value_type;
    // clang-format off
    return {FloorModParams(ov::PartialShape{8}, ov::PartialShape{8}, ov::Shape{8}, IN_ET,
                std::vector<T>{-3.2, -3.1, -3.0,  5.0,  5.1,  5.2, -1.6, 1.6},
                std::vector<T>{-3.1, -3.1, -3.1, -5.1, -5.1, -5.1,  1.7, 1.7},
                std::vector<T>{-0.1, -0.0, -3.0, -0.1, -0.0, -5.0,  0.1, 1.6})};
    // clang-format on
}

std::vector<FloorModParams> generateCombinedParamsForFloorMod() {
    const std::vector<std::vector<FloorModParams>> allTypeParams{generateParamsForFloorMod<element::Type_t::f32>(),
                                                                 generateParamsForFloorMod<element::Type_t::f16>(),
                                                                 generateParamsForFloorMod<element::Type_t::bf16>(),
                                                                 generateParamsForFloorMod<element::Type_t::i64>(),
                                                                 generateParamsForFloorMod<element::Type_t::i32>(),
                                                                 generateParamsForFloorMod<element::Type_t::i8>()};

    std::vector<FloorModParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

std::vector<FloorModParams> generateCombinedParamsForFloorModBroadcast() {
    const std::vector<std::vector<FloorModParams>> allTypeParams{
        generateParamsForFloorModBroadcast<element::Type_t::f32>(),
        generateParamsForFloorModBroadcast<element::Type_t::f16>()};

    std::vector<FloorModParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

std::vector<FloorModParams> generateCombinedParamsForFloorModScalar() {
    const std::vector<std::vector<FloorModParams>> allTypeParams{
        generateParamsForFloorModScalar<element::Type_t::f32>(),
        generateParamsForFloorModScalar<element::Type_t::f16>(),
        generateParamsForFloorModScalar<element::Type_t::bf16>(),
        generateParamsForFloorModScalar<element::Type_t::i64>(),
        generateParamsForFloorModScalar<element::Type_t::i32>(),
        generateParamsForFloorModScalar<element::Type_t::i8>(),
        generateParamsForFloorModScalar<element::Type_t::u64>(),
        generateParamsForFloorModScalar<element::Type_t::u32>(),
        generateParamsForFloorModScalar<element::Type_t::u8>()};

    std::vector<FloorModParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

std::vector<FloorModParams> generateCombinedParamsForFloorModNonIntegerDivisor() {
    return generateParamsForFloorModNonIntegerDivisor<element::Type_t::f32>();
}

INSTANTIATE_TEST_SUITE_P(smoke_FloorMod_With_Hardcoded_Refs,
                         ReferenceFloorModLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForFloorMod()),
                         ReferenceFloorModLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FloorMod_Broadcast_With_Hardcoded_Refs,
                         ReferenceFloorModLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForFloorModBroadcast()),
                         ReferenceFloorModLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FloorMod_Scalar_With_Hardcoded_Refs,
                         ReferenceFloorModLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForFloorModScalar()),
                         ReferenceFloorModLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FloorMod_NonInteger_Divisor,
                         ReferenceFloorModLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForFloorModNonIntegerDivisor()),
                         ReferenceFloorModLayerTest::getTestCaseName);

}  // namespace
