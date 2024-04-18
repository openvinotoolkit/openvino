// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/result.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"

using namespace reference_tests;
using namespace ov;

namespace {

struct ResultParams {
    template <class T>
    ResultParams(const Shape& input_shape,
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

class ReferenceResultLayerTest : public testing::TestWithParam<ResultParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto params = GetParam();
        function = CreateFunction(params.m_input_shape, params.m_input_type);
        inputData = {params.m_input_value};
        refOutData = {params.m_expected_value};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ResultParams>& obj) {
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
        auto result = std::make_shared<op::v0::Result>(in);
        return std::make_shared<ov::Model>(result, ParameterVector{in});
    }
};

TEST_P(ReferenceResultLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<ResultParams> generateParamsForResult() {
    using T = typename element_type_traits<ET>::value_type;

    std::vector<ResultParams> params{
        ResultParams(Shape{2, 2}, Shape{2, 2}, ET, ET, std::vector<T>{1, 2, 3, 5}, std::vector<T>{1, 2, 3, 5})};

    return params;
}

std::vector<ResultParams> generateCombinedParamsForResult() {
    const std::vector<std::vector<ResultParams>> allTypeParams{generateParamsForResult<element::Type_t::boolean>(),
                                                               generateParamsForResult<element::Type_t::f32>(),
                                                               generateParamsForResult<element::Type_t::f16>(),
                                                               generateParamsForResult<element::Type_t::bf16>(),
                                                               generateParamsForResult<element::Type_t::i64>(),
                                                               generateParamsForResult<element::Type_t::i32>(),
                                                               generateParamsForResult<element::Type_t::i16>(),
                                                               generateParamsForResult<element::Type_t::i8>(),
                                                               generateParamsForResult<element::Type_t::u64>(),
                                                               generateParamsForResult<element::Type_t::u32>(),
                                                               generateParamsForResult<element::Type_t::u16>(),
                                                               generateParamsForResult<element::Type_t::u8>()};

    std::vector<ResultParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Result_With_Hardcoded_Refs,
                         ReferenceResultLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForResult()),
                         ReferenceResultLayerTest::getTestCaseName);

}  // namespace
