// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/util/variable.hpp"

using namespace ov;
using namespace reference_tests;

namespace {

struct ReadValueAssignParams {
    template <class IT>
    ReadValueAssignParams(const Shape& input_shape,
                          const Shape& output_shape,
                          const element::Type& input_type,
                          const element::Type& ouput_type,
                          const std::vector<IT>& input_values,
                          const std::vector<IT>& output_values,
                          const std::string& variable_id)
        : m_input_shape(input_shape),
          m_output_shape(output_shape),
          m_input_type(input_type),
          m_output_type(ouput_type),
          m_input_data(CreateTensor(input_type, input_values)),
          m_expected_data(CreateTensor(ouput_type, output_values)),
          m_variable_id(variable_id) {}
    Shape m_input_shape;
    Shape m_output_shape;
    element::Type m_input_type;
    element::Type m_output_type;
    runtime::Tensor m_input_data;
    runtime::Tensor m_expected_data;
    std::string m_variable_id;
};

class ReferenceReadValueAssignV3LayerTest : public testing::TestWithParam<ReadValueAssignParams>,
                                            public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.m_input_shape, params.m_input_type, params.m_variable_id);
        inputData = {params.m_input_data};
        refOutData = {params.m_expected_data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ReadValueAssignParams>& obj) {
        auto params = obj.param;
        std::ostringstream result;
        result << "shape=" << params.m_input_shape << "_";
        result << "iType=" << params.m_input_type << "_";
        result << "shape=" << params.m_output_shape << "_";
        result << "oType=" << params.m_output_type;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& input_shape,
                                                 const element::Type& input_type,
                                                 const std::string variable_id) {
        auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        auto read_value = std::make_shared<op::v3::ReadValue>(in, variable_id);
        auto assign = std::make_shared<op::v3::Assign>(read_value, variable_id);
        return std::make_shared<Model>(OutputVector{assign}, ParameterVector{in});
    }
};

class ReferenceReadValueAssignV6LayerTest : public testing::TestWithParam<ReadValueAssignParams>,
                                            public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.m_input_shape, params.m_input_type, params.m_variable_id);
        inputData = {params.m_input_data};
        refOutData = {params.m_expected_data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ReadValueAssignParams>& obj) {
        auto params = obj.param;
        std::ostringstream result;
        result << "shape=" << params.m_input_shape << "_";
        result << "iType=" << params.m_input_type << "_";
        result << "shape=" << params.m_output_shape << "_";
        result << "oType=" << params.m_output_type;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& input_shape,
                                                 const element::Type& input_type,
                                                 const std::string variable_id) {
        auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        auto variable = std::make_shared<op::util::Variable>(
            op::util::VariableInfo{PartialShape::dynamic(), element::dynamic, variable_id});
        auto assign = std::make_shared<op::v6::Assign>(in, variable);
        auto read_value = std::make_shared<op::v6::ReadValue>(assign, variable);
        return std::make_shared<Model>(OutputVector{read_value},
                                       ParameterVector{in},
                                       op::util::VariableVector{variable});
    }
};

TEST_P(ReferenceReadValueAssignV3LayerTest, ReadValueAssignWithHardcodedRefs) {
    Exec();
}

TEST_P(ReferenceReadValueAssignV6LayerTest, ReadValueAssignWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<ReadValueAssignParams> generateParamsForReadValueAssign() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<ReadValueAssignParams> params{
        ReadValueAssignParams(ov::Shape{1}, ov::Shape{1}, IN_ET, IN_ET, std::vector<T>{1}, std::vector<T>{1}, "v0"),
        ReadValueAssignParams(ov::Shape{2, 2},
                              ov::Shape{2, 2},
                              IN_ET,
                              IN_ET,
                              std::vector<T>{1, 2, 3, 4},
                              std::vector<T>{1, 2, 3, 4},
                              "v0"),
        ReadValueAssignParams(ov::Shape{1, 2, 3},
                              ov::Shape{1, 2, 3},
                              IN_ET,
                              IN_ET,
                              std::vector<T>{1, 2, 3, 4, 5, 6},
                              std::vector<T>{1, 2, 3, 4, 5, 6},
                              "v0")};
    return params;
}

template <element::Type_t IN_ET>
std::vector<ReadValueAssignParams> generateParamsForReadValueAssignBoolean() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<ReadValueAssignParams> params{
        ReadValueAssignParams(ov::Shape{1}, ov::Shape{1}, IN_ET, IN_ET, std::vector<T>{true}, std::vector<T>{true}, "v0"),
        ReadValueAssignParams(ov::Shape{2, 2},
                              ov::Shape{2, 2},
                              IN_ET,
                              IN_ET,
                              std::vector<T>{true, true, false, false},
                              std::vector<T>{true, true, false, false},
                              "v0"),
        ReadValueAssignParams(ov::Shape{1, 2, 3},
                              ov::Shape{1, 2, 3},
                              IN_ET,
                              IN_ET,
                              std::vector<T>{true, false, true, false, true, false},
                              std::vector<T>{true, false, true, false, true, false},
                              "v0")};
    return params;
}

std::vector<ReadValueAssignParams> generateCombinedParamsForReadValueAssign() {
    const std::vector<std::vector<ReadValueAssignParams>> allTypeParams{
        generateParamsForReadValueAssign<element::Type_t::f64>(),
        generateParamsForReadValueAssign<element::Type_t::f32>(),
        generateParamsForReadValueAssign<element::Type_t::f16>(),
        generateParamsForReadValueAssign<element::Type_t::bf16>(),
        generateParamsForReadValueAssign<element::Type_t::i64>(),
        generateParamsForReadValueAssign<element::Type_t::i32>(),
        generateParamsForReadValueAssign<element::Type_t::i16>(),
        generateParamsForReadValueAssign<element::Type_t::i8>(),
        generateParamsForReadValueAssign<element::Type_t::i4>(),
        generateParamsForReadValueAssign<element::Type_t::u64>(),
        generateParamsForReadValueAssign<element::Type_t::u32>(),
        generateParamsForReadValueAssign<element::Type_t::u16>(),
        generateParamsForReadValueAssign<element::Type_t::u8>(),
        generateParamsForReadValueAssign<element::Type_t::u4>(),
        generateParamsForReadValueAssignBoolean<element::Type_t::boolean>()};

    std::vector<ReadValueAssignParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_ReadValue_Assign_With_Hardcoded_Refs,
                         ReferenceReadValueAssignV3LayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForReadValueAssign()),
                         ReferenceReadValueAssignV3LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReadValue_Assign_With_Hardcoded_Refs,
                         ReferenceReadValueAssignV6LayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForReadValueAssign()),
                         ReferenceReadValueAssignV6LayerTest::getTestCaseName);

}  // namespace
