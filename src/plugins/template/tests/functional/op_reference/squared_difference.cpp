// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/squared_difference.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace ov;
using namespace reference_tests;

namespace {

struct SquaredDifferenceParams {
    template <class IT>
    SquaredDifferenceParams(const PartialShape& iShape1,
                            const PartialShape& iShape2,
                            const element::Type& iType,
                            const std::vector<IT>& iValues1,
                            const std::vector<IT>& iValues2,
                            const Shape& output_shape,
                            const std::vector<IT>& oValues)
        : pshape1(iShape1),
          pshape2(iShape2),
          inType(iType),
          outType(iType),
          inputData1(CreateTensor(iType, iValues1)),
          inputData2(CreateTensor(iType, iValues2)),
          refData(CreateTensor(output_shape, iType, oValues)) {}

    PartialShape pshape1;
    PartialShape pshape2;
    element::Type inType;
    element::Type outType;
    ov::Tensor inputData1;
    ov::Tensor inputData2;
    ov::Tensor refData;
};

class ReferenceSquaredDifferenceLayerTest : public testing::TestWithParam<SquaredDifferenceParams>,
                                            public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params.pshape1, params.pshape2, params.inType, params.outType);
        inputData = {params.inputData1, params.inputData2};
        refOutData = {params.refData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<SquaredDifferenceParams>& obj) {
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
        const auto squared_difference = std::make_shared<op::v0::SquaredDifference>(in1, in2);

        return std::make_shared<Model>(NodeVector{squared_difference}, ParameterVector{in1, in2});
    }
};

class ReferenceSquaredDifferenceInPlaceLayerTest : public testing::TestWithParam<SquaredDifferenceParams>,
                                                   public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params.pshape1, params.pshape2, params.inType, params.outType);
        inputData = {params.inputData1, params.inputData2};
        refOutData = {params.refData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<SquaredDifferenceParams>& obj) {
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
        auto squared_difference = std::make_shared<op::v0::SquaredDifference>(in1, in2);
        squared_difference = std::make_shared<op::v0::SquaredDifference>(squared_difference, squared_difference);

        return std::make_shared<Model>(NodeVector{squared_difference}, ParameterVector{in1, in2});
    }
};

TEST_P(ReferenceSquaredDifferenceLayerTest, SquaredDifferenceWithHardcodedRefs) {
    Exec();
}

TEST_P(ReferenceSquaredDifferenceInPlaceLayerTest, SquaredDifferenceWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<SquaredDifferenceParams> generateParamsForSquaredDifference() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<SquaredDifferenceParams> params{
        SquaredDifferenceParams(ov::PartialShape{1, 2},
                                ov::PartialShape{1, 2},
                                IN_ET,
                                std::vector<T>{256, 56},
                                std::vector<T>{256, 56},
                                Shape{1, 2},
                                std::vector<T>{0, 0}),
        SquaredDifferenceParams(ov::PartialShape{2, 2},
                                ov::PartialShape{2, 2},
                                IN_ET,
                                std::vector<T>{256, 56, -21, -14},
                                std::vector<T>{-112, 56, 6, -8},
                                Shape{2, 2},
                                std::vector<T>{135424, 0, 729, 36}),
        SquaredDifferenceParams(ov::PartialShape{1, 2},
                                ov::PartialShape{3, 2, 2},
                                IN_ET,
                                std::vector<T>{1, 2},
                                std::vector<T>{5, 6, 7, 8, 2, 3, 1, 5, 6, 7, 1, 3},
                                Shape{3, 2, 2},
                                std::vector<T>{16, 16, 36, 36, 1, 1, 0, 9, 25, 25, 0, 1}),
        SquaredDifferenceParams(ov::PartialShape{1},
                                ov::PartialShape{1},
                                IN_ET,
                                std::vector<T>{57},
                                std::vector<T>{13},
                                Shape{1},
                                std::vector<T>{1936}),
        SquaredDifferenceParams(ov::PartialShape{2, 2},
                                ov::PartialShape{1},
                                IN_ET,
                                std::vector<T>{2, 4, 7, 8},
                                std::vector<T>{8},
                                Shape{2, 2},
                                std::vector<T>{36, 16, 1, 0})};
    return params;
}

template <element::Type_t IN_ET>
std::vector<SquaredDifferenceParams> generateParamsForSquaredDifferenceInPlace() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<SquaredDifferenceParams> params{SquaredDifferenceParams(ov::PartialShape{2, 2},
                                                                        ov::PartialShape{2, 2},
                                                                        IN_ET,
                                                                        std::vector<T>{1, 2, 3, 4},
                                                                        std::vector<T>{5, 6, 7, 8},
                                                                        Shape{2, 2},
                                                                        std::vector<T>{0, 0, 0, 0})};
    return params;
}

std::vector<SquaredDifferenceParams> generateCombinedParamsForSquaredDifference() {
    const std::vector<std::vector<SquaredDifferenceParams>> allTypeParams{
        generateParamsForSquaredDifference<element::Type_t::f32>(),
        generateParamsForSquaredDifference<element::Type_t::f16>(),
        generateParamsForSquaredDifference<element::Type_t::i64>(),
        generateParamsForSquaredDifference<element::Type_t::i32>()};

    std::vector<SquaredDifferenceParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

std::vector<SquaredDifferenceParams> generateCombinedParamsForSquaredDifferenceInPlace() {
    const std::vector<std::vector<SquaredDifferenceParams>> allTypeParams{
        generateParamsForSquaredDifferenceInPlace<element::Type_t::f32>(),
        generateParamsForSquaredDifferenceInPlace<element::Type_t::f16>(),
        generateParamsForSquaredDifferenceInPlace<element::Type_t::i64>(),
        generateParamsForSquaredDifferenceInPlace<element::Type_t::i32>()};

    std::vector<SquaredDifferenceParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_SquaredDifference_With_Hardcoded_Refs,
                         ReferenceSquaredDifferenceLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForSquaredDifference()),
                         ReferenceSquaredDifferenceLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SquaredDifferenceInPlace_With_Hardcoded_Refs,
                         ReferenceSquaredDifferenceInPlaceLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForSquaredDifferenceInPlace()),
                         ReferenceSquaredDifferenceInPlaceLayerTest::getTestCaseName);

}  // namespace
