// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/maximum.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace ov;
using namespace reference_tests;

namespace {

struct MaximumParams {
    template <class IT>
    MaximumParams(const PartialShape& iShape1,
                  const PartialShape& iShape2,
                  const element::Type& iType,
                  const std::vector<IT>& iValues1,
                  const std::vector<IT>& iValues2,
                  const std::vector<IT>& oValues)
        : pshape1(iShape1),
          pshape2(iShape2),
          inType(iType),
          outType(iType),
          inputData1(CreateTensor(iShape1.get_shape(), iType, iValues1)),
          inputData2(CreateTensor(iShape2.get_shape(), iType, iValues2)),
          refData(CreateTensor(iShape1.get_shape(), iType, oValues)) {}

    PartialShape pshape1;
    PartialShape pshape2;
    element::Type inType;
    element::Type outType;
    ov::Tensor inputData1;
    ov::Tensor inputData2;
    ov::Tensor refData;
};

class ReferenceMaximumLayerTest : public testing::TestWithParam<MaximumParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params.pshape1, params.pshape2, params.inType, params.outType);
        inputData = {params.inputData1, params.inputData2};
        refOutData = {params.refData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<MaximumParams>& obj) {
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
        const auto maximum = std::make_shared<op::v1::Maximum>(in1, in2);

        return std::make_shared<Model>(NodeVector{maximum}, ParameterVector{in1, in2});
    }
};

TEST_P(ReferenceMaximumLayerTest, MaximumWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<MaximumParams> generateParamsForMaximumFloat() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<MaximumParams> params{MaximumParams(ov::PartialShape{2, 2, 2},
                                                    ov::PartialShape{2, 2, 2},
                                                    IN_ET,
                                                    std::vector<T>{1, 8, -8, 17, -0.5, 0.5, 2, 1},
                                                    std::vector<T>{1, 2, 4, 8, 0, 0, 1, 1.5},
                                                    std::vector<T>{1, 8, 4, 17, 0, 0.5, 2, 1.5})};
    return params;
}

template <element::Type_t IN_ET>
std::vector<MaximumParams> generateParamsForMaximumInt32() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<MaximumParams> params{MaximumParams(ov::PartialShape{2, 2},
                                                    ov::PartialShape{2, 2},
                                                    IN_ET,
                                                    std::vector<T>{0x40000140, 0x40000001, -8, 17},
                                                    std::vector<T>{0x40000170, 0x40000000, 4, 8},
                                                    std::vector<T>{0x40000170, 0x40000001, 4, 17})};
    return params;
}

template <element::Type_t IN_ET>
std::vector<MaximumParams> generateParamsForMaximumInt64() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<MaximumParams> params{MaximumParams(ov::PartialShape{2, 2, 2},
                                                    ov::PartialShape{2, 2, 2},
                                                    IN_ET,
                                                    std::vector<T>{1, 8, -8, 17, -5, 67635216, 2, 17179887632},
                                                    std::vector<T>{1, 2, 4, 8, 0, 18448, 1, 28059},
                                                    std::vector<T>{1, 8, 4, 17, 0, 67635216, 2, 17179887632})};
    return params;
}

template <element::Type_t IN_ET>
std::vector<MaximumParams> generateParamsForMaximumUnsignedInt() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<MaximumParams> params{MaximumParams(ov::PartialShape{2, 2, 2},
                                                    ov::PartialShape{2, 2, 2},
                                                    IN_ET,
                                                    std::vector<T>{1, 8, 7, 17, 5, 67635216, 2, 17179887},
                                                    std::vector<T>{1, 2, 4, 8, 0, 18448, 1, 28059},
                                                    std::vector<T>{1, 8, 7, 17, 5, 67635216, 2, 17179887})};
    return params;
}

std::vector<MaximumParams> generateCombinedParamsForMaximumFloat() {
    const std::vector<std::vector<MaximumParams>> allTypeParams{generateParamsForMaximumFloat<element::Type_t::f32>(),
                                                                generateParamsForMaximumFloat<element::Type_t::f16>()};

    std::vector<MaximumParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

std::vector<MaximumParams> generateCombinedParamsForMaximumInt32() {
    const std::vector<std::vector<MaximumParams>> allTypeParams{generateParamsForMaximumInt32<element::Type_t::i32>()};

    std::vector<MaximumParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

std::vector<MaximumParams> generateCombinedParamsForMaximumInt64() {
    const std::vector<std::vector<MaximumParams>> allTypeParams{generateParamsForMaximumInt64<element::Type_t::i64>()};

    std::vector<MaximumParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

std::vector<MaximumParams> generateCombinedParamsForMaximumUnsignedInt() {
    const std::vector<std::vector<MaximumParams>> allTypeParams{
        generateParamsForMaximumUnsignedInt<element::Type_t::u64>(),
        generateParamsForMaximumUnsignedInt<element::Type_t::u32>()};

    std::vector<MaximumParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Maximum_Float_With_Hardcoded_Refs,
                         ReferenceMaximumLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForMaximumFloat()),
                         ReferenceMaximumLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Maximum_Int32_With_Hardcoded_Refs,
                         ReferenceMaximumLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForMaximumInt32()),
                         ReferenceMaximumLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Maximum_Int64_With_Hardcoded_Refs,
                         ReferenceMaximumLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForMaximumInt64()),
                         ReferenceMaximumLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Maximum_UnsignedInt_With_Hardcoded_Refs,
                         ReferenceMaximumLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForMaximumUnsignedInt()),
                         ReferenceMaximumLayerTest::getTestCaseName);

}  // namespace
