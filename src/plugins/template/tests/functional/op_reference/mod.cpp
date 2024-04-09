// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/mod.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace ov;
using namespace reference_tests;

namespace {

struct ModParams {
    template <class IT>
    ModParams(const Shape& iShape1,
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

class ReferenceModLayerTest : public testing::TestWithParam<ModParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape1, params.pshape2, params.inType, params.outType);
        inputData = {params.inputData1, params.inputData2};
        refOutData = {params.refData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ModParams>& obj) {
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
        const auto mod = std::make_shared<op::v1::Mod>(in1, in2);

        return std::make_shared<Model>(NodeVector{mod}, ParameterVector{in1, in2});
    }
};

class ReferenceModInPlaceLayerTest : public testing::TestWithParam<ModParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape1, params.pshape2, params.inType, params.outType);
        inputData = {params.inputData1, params.inputData2};
        refOutData = {params.refData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ModParams>& obj) {
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
        auto mod = std::make_shared<op::v1::Mod>(in1, in2);
        mod = std::make_shared<op::v1::Mod>(mod, mod);

        return std::make_shared<Model>(NodeVector{mod}, ParameterVector{in1, in2});
    }
};

TEST_P(ReferenceModLayerTest, ModWithHardcodedRefs) {
    Exec();
}

TEST_P(ReferenceModInPlaceLayerTest, ModWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<ModParams> generateParamsForMod() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<ModParams> params{ModParams(ov::Shape{1, 2},
                                            ov::Shape{1, 2},
                                            ov::Shape{1, 2},
                                            IN_ET,
                                            std::vector<T>{256, 56},
                                            std::vector<T>{256, 56},
                                            std::vector<T>{0, 0}),
                                  ModParams(ov::Shape{2, 2},
                                            ov::Shape{2, 2},
                                            ov::Shape{2, 2},
                                            IN_ET,
                                            std::vector<T>{256, 56, 21, 14},
                                            std::vector<T>{112, 56, 6, 8},
                                            std::vector<T>{32, 0, 3, 6}),
                                  ModParams(ov::Shape{1, 2},
                                            ov::Shape{3, 2, 2},
                                            ov::Shape{3, 2, 2},
                                            IN_ET,
                                            std::vector<T>{1, 2},
                                            std::vector<T>{5, 6, 7, 8, 2, 3, 1, 5, 6, 7, 1, 3},
                                            std::vector<T>{1, 2, 1, 2, 1, 2, 0, 2, 1, 2, 0, 2}),
                                  ModParams(ov::Shape{1},
                                            ov::Shape{1},
                                            ov::Shape{1},
                                            IN_ET,
                                            std::vector<T>{57},
                                            std::vector<T>{13},
                                            std::vector<T>{5}),
                                  ModParams(ov::Shape{2, 2},
                                            ov::Shape{1},
                                            ov::Shape{2, 2},
                                            IN_ET,
                                            std::vector<T>{2, 4, 7, 8},
                                            std::vector<T>{8},
                                            std::vector<T>{2, 4, 7, 0})};
    return params;
}

template <element::Type_t IN_ET>
std::vector<ModParams> generateParamsForModNegative() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<ModParams> params{ModParams(ov::Shape{2, 2},
                                            ov::Shape{2, 2},
                                            ov::Shape{2, 2},
                                            IN_ET,
                                            std::vector<T>{-57, -14, -12, -6},
                                            std::vector<T>{13, -7, 5, -5},
                                            std::vector<T>{-5, 0, -2, -1})};
    return params;
}

template <element::Type_t IN_ET>
std::vector<ModParams> generateParamsForModInPlace() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<ModParams> params{ModParams(ov::Shape{2, 2},
                                            ov::Shape{2, 2},
                                            ov::Shape{2, 2},
                                            IN_ET,
                                            std::vector<T>{1, 2, 3, 4},
                                            std::vector<T>{5, 6, 7, 8},
                                            std::vector<T>{0, 0, 0, 0})};
    return params;
}

std::vector<ModParams> generateCombinedParamsForMod() {
    const std::vector<std::vector<ModParams>> allTypeParams{generateParamsForMod<element::Type_t::f32>(),
                                                            generateParamsForMod<element::Type_t::f16>(),
                                                            generateParamsForMod<element::Type_t::i64>(),
                                                            generateParamsForMod<element::Type_t::i32>()};

    std::vector<ModParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

std::vector<ModParams> generateCombinedParamsForModNegative() {
    const std::vector<std::vector<ModParams>> allTypeParams{generateParamsForModNegative<element::Type_t::f32>(),
                                                            generateParamsForModNegative<element::Type_t::f16>(),
                                                            generateParamsForModNegative<element::Type_t::i64>(),
                                                            generateParamsForModNegative<element::Type_t::i32>()};

    std::vector<ModParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

std::vector<ModParams> generateCombinedParamsForModInPlace() {
    const std::vector<std::vector<ModParams>> allTypeParams{generateParamsForModInPlace<element::Type_t::f32>(),
                                                            generateParamsForModInPlace<element::Type_t::f16>(),
                                                            generateParamsForModInPlace<element::Type_t::i64>(),
                                                            generateParamsForModInPlace<element::Type_t::i32>()};

    std::vector<ModParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Mod_With_Hardcoded_Refs,
                         ReferenceModLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForMod()),
                         ReferenceModLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Mod_Negative_With_Hardcoded_Refs,
                         ReferenceModLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForModNegative()),
                         ReferenceModLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Mod_InPlace_With_Hardcoded_Refs,
                         ReferenceModInPlaceLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForModInPlace()),
                         ReferenceModInPlaceLayerTest::getTestCaseName);

}  // namespace
