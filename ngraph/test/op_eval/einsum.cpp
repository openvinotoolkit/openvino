// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/op/einsum.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/validation_util.hpp"
#include "runtime/backend.hpp"
#include "util/engine/interpreter_engine.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_tools.hpp"


using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

template <typename T>
static void aux_einsum_test(const std::vector<std::vector<T>>& inputs, const std::vector<Shape>& input_shapes,
    const std::string& equation, const std::vector<T>& expected_result, const Shape& expected_shape)
{
    NGRAPH_CHECK(inputs.size() == input_shapes.size());
    OutputVector output_vector;
    ParameterVector param_vector;
    for (const auto& input_shape : input_shapes) {
        auto param = make_shared<op::Parameter>(element::from<T>(), input_shape);
        output_vector.push_back(param);
        param_vector.push_back(param);
    }

    auto einsum = make_shared<op::v7::Einsum>(output_vector, equation);
    auto fun = make_shared<Function>(OutputVector{einsum}, param_vector);

    auto test_case = test::TestCase<ngraph::test::INTERPRETER_Engine>(fun);
    for (size_t ind = 0; ind < inputs.size(); ++ind) {
        test_case.add_input<T>(input_shapes[ind], inputs[ind]);
    }
    test_case.add_expected_output<T>(expected_shape, expected_result);
    test_case.run();
}

TEST(op_eval, einsum_no_reduction)
{
    std::string equation = "ab,cd->abcd";
    std::vector<float> input1{1.0f, 2.0f};
    Shape input1_shape{1, 2};
    std::vector<float> input2{3.0f, 4.0f, 5.0f, 6.0f,
                              7.0f, 8.0f, 9.0f, 10.0f,
                              11.0f, 12.0f, 13.0f, 14.0f};
    Shape input2_shape{3, 4};
    std::vector<float> expected_result{
        3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
        6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f
    };
    Shape expected_shape{1, 2, 3, 4};

    aux_einsum_test(
        {input1, input2}, {input1_shape, input2_shape}, equation, expected_result, expected_shape);
}

TEST(op_eval, einsum_transpose)
{
    std::string equation = "ijk->kij";
    std::vector<float> input1{1.0f, 2.0f, 3.0f,
                              4.0f, 5.0f, 6.0f};
    Shape input1_shape{1, 2, 3};
    std::vector<float> expected_result{1.0f, 4.0f,
                                       2.0f, 5.0f,
                                       3.0f, 6.0f};
    Shape expected_shape{3, 1, 2};

    aux_einsum_test(
        {input1}, {input1_shape}, equation, expected_result, expected_shape);
}

TEST(op_eval, einsum_reduce)
{
    std::string equation = "ab->a";
    std::vector<float> input1{1.0f, 2.0f, 3.0f,
                              4.0f, 5.0f, 6.0f};
    Shape input1_shape{2, 3};
    std::vector<float> expected_result{6.0f, 15.0f};
    Shape expected_shape{2};

    aux_einsum_test({input1}, {input1_shape}, equation, expected_result, expected_shape);
}

TEST(op_eval, einsum_matrix_multiplication)
{
    std::string equation = "ab,bc->ac";
    std::vector<float> input1{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Shape input1_shape{2, 3};
    std::vector<float> input2{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Shape input2_shape{3, 2};
    std::vector<float> expected_result{22.0f, 28.0f, 49.0f, 64.0f};
    Shape expected_shape{2, 2};

    aux_einsum_test(
        {input1, input2}, {input1_shape, input2_shape}, equation, expected_result, expected_shape);
}

TEST(op_eval, einsum_multiple_multiplication)
{
    std::string equation = "ab,bcd,bc->ca";
    std::vector<float> input1{1.0, 3.0, 2.0, 7.0,
                              5.0, 6.0, 0.0, 1.0};
    Shape input1_shape{2, 4};
    std::vector<float> input2{1.0, 2.0, 3.0,
                              4.0, 5.0, 6.0,
                              5.0, 7.0, 3.0,
                              7.0, 9.0, 1.0};
    Shape input2_shape{4, 3, 1};
    std::vector<float> input3{4.0, 3.0, 1.0, 
                              6.0, 4.0, 2.0,
                              2.0, 5.0, 3.0,
                              1.0, 9.0, 4.0};
    Shape input3_shape{4, 3};

    std::vector<float> expected_result{145.0, 171.0,
                                       703.0, 231.0,
                                       85.0, 91.0};
    Shape expected_shape{3, 2};

    aux_einsum_test(
        {input1, input2, input3},
        {input1_shape, input2_shape, input3_shape},
        equation, expected_result, expected_shape);
}

TEST(op_eval, einsum_ellipsis_one_input_reduction)
{
    std::string equation = "a...->...";
    std::vector<float> input1{1.0, 3.0, 2.0, 7.0, 5.0, 6.0,
                              3.0, 5.0, 2.0, 1.0, 0.0, 7.0};
    Shape input1_shape{2, 2, 3};

    std::vector<float> expected_result{4.0, 8.0, 4.0, 8.0, 5.0, 13.0};
    Shape expected_shape{2, 3};

    aux_einsum_test({input1},
                    {input1_shape},
                    equation,
                    expected_result,
                    expected_shape);
}

TEST(op_eval, einsum_ellipsis_one_input_transpose)
{
    std::string equation = "a...->...a";
    std::vector<float> input1{1.0, 3.0, 2.0, 7.0, 5.0, 6.0,
                              3.0, 5.0, 2.0, 1.0, 0.0, 7.0};
    Shape input1_shape{2, 2, 3};

    std::vector<float> expected_result{1.0, 3.0, 3.0, 5.0, 2.0, 2.0,
                                       7.0, 1.0, 5.0, 0.0, 6.0, 7.0,};
    Shape expected_shape{2, 3, 2};

    aux_einsum_test({input1}, {input1_shape}, equation, expected_result, expected_shape);
}

TEST(op_eval, einsum_ellipsis_mul_by_1dscalar)
{
    std::string equation = "ab...,...->ab...";
    std::vector<float> input1{1.0, 3.0, 2.0, 7.0, 5.0, 6.0,
                              3.0, 5.0, 2.0, 1.0, 0.0, 7.0};
    Shape input1_shape{2, 2, 3};
    std::vector<float> input2{0.5};
    Shape input2_shape{1};

    std::vector<float> expected_result{0.5, 1.5, 1.0, 3.5, 2.5, 3.0,
                                       1.5, 2.5, 1.0, 0.5, 0.0, 3.5};
    Shape expected_shape{2, 2, 3};

    aux_einsum_test({input1, input2}, {input1_shape, input2_shape}, equation, expected_result, expected_shape);
}

TEST(op_eval, einsum_ellipsis_complex_mul)
{
    std::string equation = "a...j,j...->a...";
    std::vector<float> input1{1.0, 3.0, 2.0, 7.0, 5.0, 6.0, 3.0, 5.0, 2.0, 1.0, 0.0, 7.0};
    Shape input1_shape{1, 1, 4, 3};
    std::vector<float> input2{3.0, 1.0, 6.0, 2.0, 3.0, 10.0, 9.0, 8.0, 2.0, 9.0, 3.0, 2.0,
                              4.0, 2.0, 3.0, 1.0, 9.0, 1.0, 11.0, 4.0, 7.0, 2.0, 3.0, 1.0};
    Shape input2_shape{3, 4, 2, 1};

    std::vector<float> expected_result{27., 85., 37., 66., 30., 58., 50., 8.,
                                       37., 123., 55., 83., 16., 48., 24., 30.,
                                       29., 83., 43., 52., 20., 92., 44., 24.,
                                       24., 96., 48., 30., 13., 67., 31., 15.};
    Shape expected_shape{1, 4, 2, 4};

    aux_einsum_test(
        {input1, input2}, {input1_shape, input2_shape}, equation, expected_result, expected_shape);
}

TEST(op_eval, einsum_diagonal)
{
    std::string equation = "kii->ki";
    std::vector<float> input1{1.0f, 2.0f, 3.0f,
                              4.0f, 5.0f, 6.0f,
                              7.0f, 8.0f, 9.0f};
    Shape input1_shape{1, 3, 3};
    std::vector<float> expected_result{1.0f, 5.0f, 9.0f};
    Shape expected_shape{1, 3};

    aux_einsum_test({input1}, {input1_shape}, equation, expected_result, expected_shape);
}

TEST(op_eval, einsum_diagonal_with_matmul)
{
    std::string equation = "abbac,bad->ad";
    std::vector<float> input1{
        4.0, 2.0, 5.0, 4.0, 5.0, 5.0, 1.0, 1.0, 3.0, 3.0, 1.0, 1.0, 2.0, 2.0, 4.0, 1.0,
        3.0, 4.0, 4.0, 5.0, 1.0, 3.0, 1.0, 3.0, 1.0, 4.0, 3.0, 5.0, 4.0, 4.0, 5.0, 4.0, 4.0,
        5.0, 4.0, 2.0, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0, 4.0, 3.0, 4.0, 2.0, 2.0, 1.0, 1.0, 2.0,
        3.0, 1.0, 1.0, 4.0, 2.0, 3.0, 1.0, 3.0, 4.0, 2.0, 5.0, 5.0, 3.0, 4.0, 3.0, 4.0, 5.0,
        4.0, 4.0, 5.0, 1.0, 3.0, 4.0, 4.0, 5.0, 3.0, 1.0, 3.0, 2.0, 5.0, 3.0, 2.0, 5.0, 4.0,
        4.0, 2.0, 4.0, 4.0, 1.0, 4.0, 4.0, 5.0, 4.0, 4.0, 4.0, 2.0, 3.0, 3.0, 4.0, 2.0, 4.0,
        2.0, 5.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0, 5.0, 1.0, 1.0,
        2.0, 1.0, 4.0, 5.0, 3.0, 4.0, 1.0, 3.0, 3.0, 1.0, 3.0, 2.0, 4.0, 5.0, 1.0, 1.0, 5.0,
        4.0, 5.0, 2.0, 2.0, 3.0, 3.0, 1.0, 2.0, 4.0};
    Shape input1_shape{2, 3, 3, 2, 4};
    std::vector<float> input2{1.0, 4.0, 4.0, 5.0, 3.0, 3.0};
    Shape input2_shape{3, 2, 1};

    std::vector<float> expected_result{123, 129};
    Shape expected_shape{2, 1};

    aux_einsum_test({input1, input2}, {input1_shape, input2_shape}, equation, expected_result, expected_shape);
}
