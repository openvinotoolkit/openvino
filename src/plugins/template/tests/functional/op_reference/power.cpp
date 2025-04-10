// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/power.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace ov;
using namespace reference_tests;

namespace {

struct PowerParams {
    template <class IT>
    PowerParams(const Shape& iShape1,
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

class ReferencePowerLayerTest : public testing::TestWithParam<PowerParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape1, params.pshape2, params.inType, params.outType);
        inputData = {params.inputData1, params.inputData2};
        refOutData = {params.refData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<PowerParams>& obj) {
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
        const auto power = std::make_shared<op::v1::Power>(in1, in2);

        return std::make_shared<Model>(NodeVector{power}, ParameterVector{in1, in2});
    }
};

TEST_P(ReferencePowerLayerTest, PowerWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<PowerParams> generateParamsForPower() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<PowerParams> params{
        PowerParams(ov::Shape{2, 2},
                    ov::Shape{2, 2},
                    ov::Shape{2, 2},
                    IN_ET,
                    std::vector<T>{1, 2, 3, 5},
                    std::vector<T>{2, 0, 6, 3},
                    std::vector<T>{1, 1, 729, 125}),
        PowerParams(ov::Shape{2, 1, 5},
                    ov::Shape{2, 1},
                    ov::Shape{2, 2, 5},
                    IN_ET,
                    std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                    std::vector<T>{1, 2},
                    std::vector<T>{1, 2, 3, 4, 5, 1, 4, 9, 16, 25, 6, 7, 8, 9, 10, 36, 49, 64, 81, 100}),
        PowerParams(ov::Shape{1},
                    ov::Shape{1},
                    ov::Shape{1},
                    IN_ET,
                    std::vector<T>{2},
                    std::vector<T>{3},
                    std::vector<T>{8}),
        PowerParams(ov::Shape{2, 2},
                    ov::Shape{1},
                    ov::Shape{2, 2},
                    IN_ET,
                    std::vector<T>{2, 3, 4, 5},
                    std::vector<T>{2},
                    std::vector<T>{4, 9, 16, 25})};
    return params;
}

std::vector<PowerParams> generateCombinedParamsForPower() {
    const std::vector<std::vector<PowerParams>> allTypeParams{generateParamsForPower<element::Type_t::f32>(),
                                                              generateParamsForPower<element::Type_t::f16>(),
                                                              generateParamsForPower<element::Type_t::bf16>(),
                                                              generateParamsForPower<element::Type_t::i64>(),
                                                              generateParamsForPower<element::Type_t::i32>(),
                                                              generateParamsForPower<element::Type_t::u64>(),
                                                              generateParamsForPower<element::Type_t::u32>()};

    std::vector<PowerParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Power_With_Hardcoded_Refs,
                         ReferencePowerLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForPower()),
                         ReferencePowerLayerTest::getTestCaseName);

}  // namespace
