// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/matmul.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"

using namespace reference_tests;
using namespace ov;

namespace {

struct MatMulParams {
    template <class T>
    MatMulParams(const Shape& input_shape1,
                 const Shape& input_shape2,
                 const Shape& expected_shape,
                 const element::Type_t& input_type1,
                 const element::Type_t& input_type2,
                 const element::Type_t& expected_type,
                 const std::vector<T>& input_value1,
                 const std::vector<T>& input_value2,
                 const std::vector<T>& expected_value,
                 const bool& transpose1,
                 const bool& transpose2,
                 const bool& use_constant) {
        m_input_shape1 = input_shape1;
        m_input_shape2 = input_shape2;
        m_expected_shape = expected_shape;
        m_input_type1 = input_type1;
        m_input_type2 = input_type2;
        m_expected_type = expected_type;
        m_input_value1 = CreateTensor(input_shape1, input_type1, input_value1);
        m_input_value2 = CreateTensor(input_shape2, input_type2, input_value2);
        m_expected_value = CreateTensor(expected_shape, expected_type, expected_value);
        m_transpose1 = transpose1;
        m_transpose2 = transpose2;
        m_use_constant = use_constant;
    }

    template <class T>
    MatMulParams(const Shape& input_shape1,
                 const Shape& input_shape2,
                 const Shape& expected_shape,
                 const element::Type_t& input_type1,
                 const element::Type_t& input_type2,
                 const element::Type_t& expected_type,
                 const T& input_value_step,
                 const std::vector<T>& expected_value,
                 const bool& transpose1,
                 const bool& transpose2,
                 const bool& use_constant) {
        m_input_shape1 = input_shape1;
        m_input_shape2 = input_shape2;
        m_expected_shape = expected_shape;
        m_input_type1 = input_type1;
        m_input_type2 = input_type2;
        m_expected_type = expected_type;
        std::vector<T> input_value1(shape_size(input_shape1));
        std::vector<T> input_value2(shape_size(input_shape2));
        std::iota(std::begin(input_value1), std::end(input_value1), input_value_step);
        std::iota(std::begin(input_value2), std::end(input_value2), input_value_step);
        m_input_value1 = CreateTensor(input_shape1, input_type1, input_value1);
        m_input_value2 = CreateTensor(input_shape2, input_type2, input_value2);
        m_expected_value = CreateTensor(expected_shape, expected_type, expected_value);
        m_transpose1 = transpose1;
        m_transpose2 = transpose2;
        m_use_constant = use_constant;
    }

    Shape m_input_shape1;
    Shape m_input_shape2;
    Shape m_expected_shape;
    element::Type_t m_input_type1;
    element::Type_t m_input_type2;
    element::Type_t m_expected_type;
    ov::Tensor m_input_value1;
    ov::Tensor m_input_value2;
    ov::Tensor m_expected_value;
    bool m_transpose1;
    bool m_transpose2;
    bool m_use_constant;
};

class ReferenceMatMulLayerTest : public testing::TestWithParam<MatMulParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        if (params.m_use_constant) {
            function = CreateFunctionWithConst(params);
            inputData = {params.m_input_value1};
        } else {
            function = CreateFunctionWithParam(params);
            inputData = {params.m_input_value1, params.m_input_value2};
        }

        refOutData = {params.m_expected_value};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<MatMulParams>& obj) {
        const auto param = obj.param;
        std::ostringstream result;

        result << "input_shape1=" << param.m_input_shape1 << "; ";
        result << "input_shape2=" << param.m_input_shape2 << "; ";
        result << "output_shape=" << param.m_expected_shape << "; ";
        result << "input_type1=" << param.m_input_type1 << "; ";
        result << "input_type2=" << param.m_input_type2 << "; ";
        result << "output_type=" << param.m_expected_type << "; ";
        result << "transpose1=" << param.m_transpose1 << "; ";
        result << "transpose2=" << param.m_transpose2 << "; ";
        result << "use_constant=" << param.m_use_constant;

        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunctionWithParam(MatMulParams& p) {
        auto in1 = std::make_shared<op::v0::Parameter>(p.m_input_type1, p.m_input_shape1);
        auto in2 = std::make_shared<op::v0::Parameter>(p.m_input_type2, p.m_input_shape2);
        auto matmul = std::make_shared<op::v0::MatMul>(in1, in2, p.m_transpose1, p.m_transpose2);

        return std::make_shared<ov::Model>(matmul, ParameterVector{in1, in2});
    }

    static std::shared_ptr<Model> CreateFunctionWithConst(MatMulParams& p) {
        auto in1 = std::make_shared<op::v0::Parameter>(p.m_input_type1, p.m_input_shape1);
        auto in2 = std::make_shared<op::v0::Constant>(p.m_input_type2, p.m_input_shape2, p.m_input_value2.data());
        auto matmul = std::make_shared<op::v0::MatMul>(in1, in2, p.m_transpose1, p.m_transpose2);

        return std::make_shared<ov::Model>(matmul, ParameterVector{in1});
    }
};

TEST_P(ReferenceMatMulLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<MatMulParams> generateParamsForMatMul() {
    using T = typename element_type_traits<ET>::value_type;

    std::vector<MatMulParams> params{
        // matmul_2x2_2x2
        MatMulParams(Shape{2, 2},
                     Shape{2, 2},
                     Shape{2, 2},
                     ET,
                     ET,
                     ET,
                     std::vector<T>{1, 2, 3, 4},
                     std::vector<T>{5, 6, 7, 8},
                     std::vector<T>{19, 22, 43, 50},
                     false,
                     false,
                     false),
        // matmul_2x3_3x3
        MatMulParams(Shape{2, 3},
                     Shape{3, 3},
                     Shape{2, 3},
                     ET,
                     ET,
                     ET,
                     std::vector<T>{1, 2, 3, 4, 5, 6},
                     std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9},
                     std::vector<T>{30, 36, 42, 66, 81, 96},
                     false,
                     false,
                     false),
        // matmul_3x2_3x3_transpose
        MatMulParams(Shape{3, 2},
                     Shape{3, 3},
                     Shape{2, 3},
                     ET,
                     ET,
                     ET,
                     std::vector<T>{1, 4, 2, 5, 3, 6},
                     std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9},
                     std::vector<T>{30, 36, 42, 66, 81, 96},
                     true,
                     false,
                     false),
        // matmul_3x2_2x3_transpose
        MatMulParams(Shape{3, 2},
                     Shape{2, 3},
                     Shape{2, 2},
                     ET,
                     ET,
                     ET,
                     std::vector<T>{1, 4, 2, 5, 3, 6},
                     std::vector<T>{1, 3, 5, 2, 4, 6},
                     std::vector<T>{22, 28, 49, 64},
                     true,
                     true,
                     false),
        // matmul_2x3x2_3x3_transpose
        MatMulParams(Shape{2, 3, 2},
                     Shape{3, 3},
                     Shape{2, 2, 3},
                     ET,
                     ET,
                     ET,
                     std::vector<T>{1, 4, 2, 5, 3, 6, 3, 2, 1, 4, 5, 6},
                     std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9},
                     std::vector<T>{30, 36, 42, 66, 81, 96, 42, 51, 60, 60, 72, 84},
                     true,
                     false,
                     false),
        // matmul_2_2
        MatMulParams(Shape{2},
                     Shape{2},
                     Shape{},
                     ET,
                     ET,
                     ET,
                     std::vector<T>{1, 2},
                     std::vector<T>{1, 2},
                     std::vector<T>{5},
                     false,
                     false,
                     false),
        // matmul_3_x_3_false_false_param
        MatMulParams(Shape{3},
                     Shape{3},
                     Shape{},
                     ET,
                     ET,
                     ET,
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{14},
                     false,
                     false,
                     false),
        // matmul_3_x_3_true_true_param
        MatMulParams(Shape{3},
                     Shape{3},
                     Shape{},
                     ET,
                     ET,
                     ET,
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{14},
                     true,
                     true,
                     false),
        // matmul_3_x_3_false_false_const
        MatMulParams(Shape{3},
                     Shape{3},
                     Shape{},
                     ET,
                     ET,
                     ET,
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{14},
                     false,
                     false,
                     true),
        // matmul_3_x_3_true_true_const
        MatMulParams(Shape{3},
                     Shape{3},
                     Shape{},
                     ET,
                     ET,
                     ET,
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{14},
                     true,
                     true,
                     true),
        // matmul_1_3_x_3_false_false_param
        MatMulParams(Shape{1, 3},
                     Shape{3},
                     Shape{1},
                     ET,
                     ET,
                     ET,
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{14},
                     false,
                     false,
                     false),
        // matmul_1_3_x_3_false_false_const
        MatMulParams(Shape{1, 3},
                     Shape{3},
                     Shape{1},
                     ET,
                     ET,
                     ET,
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{14},
                     false,
                     false,
                     true),
        // matmul_3_1_x_3_true_false_param
        MatMulParams(Shape{3, 1},
                     Shape{3},
                     Shape{1},
                     ET,
                     ET,
                     ET,
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{14},
                     true,
                     false,
                     false),
        // matmul_3_1_x_3_true_false_const
        MatMulParams(Shape{3, 1},
                     Shape{3},
                     Shape{1},
                     ET,
                     ET,
                     ET,
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{14},
                     true,
                     false,
                     true),
        // matmul_3_x_3_1_false_false_param
        MatMulParams(Shape{3},
                     Shape{3, 1},
                     Shape{1},
                     ET,
                     ET,
                     ET,
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{14},
                     false,
                     false,
                     false),
        // matmul_3_x_3_1_false_false_const
        MatMulParams(Shape{3},
                     Shape{3, 1},
                     Shape{1},
                     ET,
                     ET,
                     ET,
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{14},
                     false,
                     false,
                     true),
        // matmul_3_x_1_3_false_true_param
        MatMulParams(Shape{3},
                     Shape{1, 3},
                     Shape{1},
                     ET,
                     ET,
                     ET,
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{14},
                     false,
                     true,
                     false),
        // matmul_3_x_1_3_false_true_const
        MatMulParams(Shape{3},
                     Shape{1, 3},
                     Shape{1},
                     ET,
                     ET,
                     ET,
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{14},
                     false,
                     true,
                     true),
        // matmul_3_x_1_3_true_true_param
        MatMulParams(Shape{3},
                     Shape{1, 3},
                     Shape{1},
                     ET,
                     ET,
                     ET,
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{14},
                     true,
                     true,
                     false),
        // matmul_3_x_1_3_true_true_const
        MatMulParams(Shape{3},
                     Shape{1, 3},
                     Shape{1},
                     ET,
                     ET,
                     ET,
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{14},
                     true,
                     true,
                     true),
        // matmul_1_1_3_x_3_false_false_param
        MatMulParams(Shape{1, 1, 3},
                     Shape{3},
                     Shape{1, 1},
                     ET,
                     ET,
                     ET,
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{14},
                     false,
                     false,
                     false),
        // matmul_1_1_3_x_3_false_false_const
        MatMulParams(Shape{1, 1, 3},
                     Shape{3},
                     Shape{1, 1},
                     ET,
                     ET,
                     ET,
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{14},
                     false,
                     false,
                     true),
        // matmul_1_3_1_x_3_true_false_param
        MatMulParams(Shape{1, 3, 1},
                     Shape{3},
                     Shape{1, 1},
                     ET,
                     ET,
                     ET,
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{14},
                     true,
                     false,
                     false),
        // matmul_1_3_1_x_3_true_false_const
        MatMulParams(Shape{1, 3, 1},
                     Shape{3},
                     Shape{1, 1},
                     ET,
                     ET,
                     ET,
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{14},
                     true,
                     false,
                     true),
        // matmul_3_x_1_3_1_false_false_param
        MatMulParams(Shape{3},
                     Shape{1, 3, 1},
                     Shape{1, 1},
                     ET,
                     ET,
                     ET,
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{14},
                     false,
                     false,
                     false),
        // matmul_3_x_1_3_1_false_false_const
        MatMulParams(Shape{3},
                     Shape{1, 3, 1},
                     Shape{1, 1},
                     ET,
                     ET,
                     ET,
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{14},
                     false,
                     false,
                     true),
        // matmul_3_x_1_1_3_false_true_param
        MatMulParams(Shape{3},
                     Shape{1, 1, 3},
                     Shape{1, 1},
                     ET,
                     ET,
                     ET,
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{14},
                     false,
                     true,
                     false),
        // matmul_3_x_1_1_3_false_true_const
        MatMulParams(Shape{3},
                     Shape{1, 1, 3},
                     Shape{1, 1},
                     ET,
                     ET,
                     ET,
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{1, 2, 3},
                     std::vector<T>{14},
                     false,
                     true,
                     true),
    };

    return params;
}

template <element::Type_t ET>
std::vector<MatMulParams> generateParamsForMatMulWithGeneratedInput() {
    using T = typename element_type_traits<ET>::value_type;

    std::vector<MatMulParams> params{
        // matmul_2x2x3_2x1x3_transpose
        MatMulParams(Shape{2, 2, 3},
                     Shape{2, 1, 3},
                     Shape{2, 2, 1},
                     ET,
                     ET,
                     ET,
                     static_cast<T>(1),
                     std::vector<T>{14, 32, 122, 167},
                     false,
                     true,
                     false),
        // matmul_2x2x3_2x3x1
        MatMulParams(Shape{2, 2, 3},
                     Shape{2, 3, 1},
                     Shape{2, 2, 1},
                     ET,
                     ET,
                     ET,
                     static_cast<T>(1),
                     std::vector<T>{14, 32, 122, 167},
                     false,
                     false,
                     false),
        // matmul_1x2x3_1x4x3x2
        MatMulParams(Shape{1, 2, 3},
                     Shape{1, 4, 3, 2},
                     Shape{1, 4, 2, 2},
                     ET,
                     ET,
                     ET,
                     static_cast<T>(0),
                     std::vector<T>{10, 13, 28, 40, 28, 31, 100, 112, 46, 49, 172, 184, 64, 67, 244, 256},
                     false,
                     false,
                     false),
        // matmul_2_2_1_3_x_3_false_false_param
        MatMulParams(Shape{2, 2, 1, 3},
                     Shape{3},
                     Shape{2, 2, 1},
                     ET,
                     ET,
                     ET,
                     static_cast<T>(0),
                     std::vector<T>{5, 14, 23, 32},
                     false,
                     false,
                     false),
        // matmul_2_2_1_3_x_3_false_false_const
        MatMulParams(Shape{2, 2, 1, 3},
                     Shape{3},
                     Shape{2, 2, 1},
                     ET,
                     ET,
                     ET,
                     static_cast<T>(0),
                     std::vector<T>{5, 14, 23, 32},
                     false,
                     false,
                     true),
        // matmul_3_x_2_2_3_1_false_false_param
        MatMulParams(Shape{3},
                     Shape{2, 2, 3, 1},
                     Shape{2, 2, 1},
                     ET,
                     ET,
                     ET,
                     static_cast<T>(0),
                     std::vector<T>{5, 14, 23, 32},
                     false,
                     false,
                     false),
        // matmul_3_x_2_2_3_1_false_false_const
        MatMulParams(Shape{3},
                     Shape{2, 2, 3, 1},
                     Shape{2, 2, 1},
                     ET,
                     ET,
                     ET,
                     static_cast<T>(0),
                     std::vector<T>{5, 14, 23, 32},
                     false,
                     false,
                     true),
    };

    return params;
}

template <element::Type_t ET>
std::vector<MatMulParams> generateParamsForMatMulWithSameBatchSize() {
    using T = typename element_type_traits<ET>::value_type;

    const auto input0_shapes = Shape{3, 2, 2, 2};
    const auto input1_shapes = Shape{3, 2, 2, 1};
    std::vector<T> input0_data(shape_size(input0_shapes));
    std::vector<T> input1_data(shape_size(input1_shapes));
    std::iota(input0_data.begin(), input0_data.end(), static_cast<T>(1));
    std::iota(input1_data.begin(), input1_data.end(), static_cast<T>(0));

    return std::vector<MatMulParams>{
        MatMulParams(input0_shapes,
                     input1_shapes,
                     Shape{3, 2, 2, 1},
                     ET,
                     ET,
                     ET,
                     input0_data,
                     input1_data,
                     std::vector<T>{2, 4, 28, 38, 86, 104, 176, 202, 298, 332, 452, 494},
                     false,
                     false,
                     false),
    };
}

std::vector<MatMulParams> generateCombinedParamsForMatMul() {
    const std::vector<std::vector<MatMulParams>> allTypeParams{
        generateParamsForMatMul<element::Type_t::f32>(),
        generateParamsForMatMul<element::Type_t::f16>(),
        generateParamsForMatMul<element::Type_t::i64>(),
        generateParamsForMatMul<element::Type_t::i32>(),
        generateParamsForMatMul<element::Type_t::u64>(),
        generateParamsForMatMul<element::Type_t::u32>(),
        generateParamsForMatMulWithGeneratedInput<element::Type_t::f32>(),
        generateParamsForMatMulWithGeneratedInput<element::Type_t::i64>(),
        generateParamsForMatMulWithGeneratedInput<element::Type_t::i32>(),
        generateParamsForMatMulWithSameBatchSize<element::Type_t::f32>(),
        generateParamsForMatMulWithSameBatchSize<element::Type_t::i64>(),
    };

    std::vector<MatMulParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_MatMul_With_Hardcoded_Refs,
                         ReferenceMatMulLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForMatMul()),
                         ReferenceMatMulLayerTest::getTestCaseName);

}  // namespace
