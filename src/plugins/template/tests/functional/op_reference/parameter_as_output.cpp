// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

using namespace reference_tests;
using namespace ov;

namespace {

struct ParameterParams {
    template <class T>
    ParameterParams(const Shape& input_shape,
                    const Shape& expected_shape,
                    const element::Type& input_type,
                    const element::Type& expected_type,
                    const std::vector<T>& input_value,
                    const std::vector<T>& expected_value)
        : m_input_shape(input_shape),
          m_expected_shape(expected_shape),
          m_input_type(input_type),
          m_expected_type(expected_type),
          m_input_value(CreateTensor(input_shape, input_type, input_value)),
          m_expected_value(CreateTensor(expected_shape, expected_type, expected_value)) {}

    Shape m_input_shape;
    Shape m_expected_shape;
    element::Type m_input_type;
    element::Type m_expected_type;
    ov::Tensor m_input_value;
    ov::Tensor m_expected_value;
};

class ReferenceParameterLayerTest : public testing::TestWithParam<ParameterParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto params = GetParam();
        function = CreateFunction(params.m_input_shape, params.m_input_type);
        inputData = {params.m_input_value};
        refOutData = {params.m_expected_value};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ParameterParams>& obj) {
        const auto param = obj.param;
        std::ostringstream result;

        result << "input_shape=" << param.m_input_shape << "; ";
        result << "output_shape=" << param.m_expected_shape << "; ";
        result << "input_type=" << param.m_input_type << "; ";
        result << "output_type=" << param.m_expected_type;

        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& input_shape, const element::Type_t& input_type) {
        auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        return std::make_shared<ov::Model>(in, ParameterVector{in});
    }
};

TEST_P(ReferenceParameterLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<ParameterParams> generateParamsForParameter() {
    using T = typename element_type_traits<ET>::value_type;

    std::vector<ParameterParams> params{ParameterParams(Shape{3, 4},
                                                        Shape{3, 4},
                                                        ET,
                                                        ET,
                                                        std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                                                        std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11})};

    return params;
}

std::vector<ParameterParams> generateCombinedParamsForParameter() {
    const std::vector<std::vector<ParameterParams>> allTypeParams{
        generateParamsForParameter<element::Type_t::boolean>(),
        generateParamsForParameter<element::Type_t::f32>(),
        generateParamsForParameter<element::Type_t::f16>(),
        generateParamsForParameter<element::Type_t::bf16>(),
        generateParamsForParameter<element::Type_t::i64>(),
        generateParamsForParameter<element::Type_t::i32>(),
        generateParamsForParameter<element::Type_t::i16>(),
        generateParamsForParameter<element::Type_t::i8>(),
        generateParamsForParameter<element::Type_t::u64>(),
        generateParamsForParameter<element::Type_t::u32>(),
        generateParamsForParameter<element::Type_t::u16>(),
        generateParamsForParameter<element::Type_t::u8>()};

    std::vector<ParameterParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Parameter_With_Hardcoded_Refs,
                         ReferenceParameterLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForParameter()),
                         ReferenceParameterLayerTest::getTestCaseName);

}  // namespace
