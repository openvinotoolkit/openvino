// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "runtime/backend.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, concat_negative_axis)
{
    auto pshape_a = PartialShape::dynamic();
    auto A = make_shared<op::Parameter>(element::f32, pshape_a);
    auto pshape_b = PartialShape::dynamic();
    auto B = make_shared<op::Parameter>(element::f32, pshape_b);
    auto pshape_c = PartialShape::dynamic();
    auto C = make_shared<op::Parameter>(element::f32, pshape_c);
    auto pshape_r = PartialShape::dynamic();
    auto f = make_shared<Function>(make_shared<op::Concat>(NodeVector{A, B, C}, -1),
                                   ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, Shape{2, 2});
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::f32, Shape{2, 3});
    copy_data(b, vector<float>{1, 2, 4, 8, 16, 32});
    auto c = backend->create_tensor(element::f32, Shape{2, 3});
    copy_data(c, vector<float>{2, 3, 5, 7, 11, 13});
    auto result = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b, c});
    ASSERT_EQ(result->get_shape(), (Shape{2, 8}));
    EXPECT_TRUE(
        test::all_close_f((vector<float>{2, 4, 1, 2, 4, 2, 3, 5, 8, 16, 8, 16, 32, 7, 11, 13}),
                          read_vector<float>(result)));
}
NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_colwise)
{
    Shape shape_a{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 3};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{2, 3};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{2, 8};
    auto f = make_shared<Function>(make_shared<op::Concat>(NodeVector{A, B, C}, 1),
                                   ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{1, 2, 4, 8, 16, 32});
    auto c = backend->create_tensor(element::f32, shape_c);
    copy_data(c, vector<float>{2, 3, 5, 7, 11, 13});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b, c});
    EXPECT_TRUE(
        test::all_close_f((vector<float>{2, 4, 1, 2, 4, 2, 3, 5, 8, 16, 8, 16, 32, 7, 11, 13}),
                          read_vector<float>(result),
                          MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_rowwise)
{
    Shape shape_a{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{3, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{3, 2};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{8, 2};
    auto f = make_shared<Function>(make_shared<op::Concat>(NodeVector{A, B, C}, 0),
                                   ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{1, 2, 4, 8, 16, 32});
    auto c = backend->create_tensor(element::f32, shape_c);
    copy_data(c, vector<float>{2, 3, 5, 7, 11, 13});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b, c});
    EXPECT_TRUE(
        test::all_close_f((vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13}),
                          read_vector<float>(result),
                          MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_int64)
{
    Shape shape_a{2, 2};
    auto A = make_shared<op::Parameter>(element::i64, shape_a);
    Shape shape_b{3, 2};
    auto B = make_shared<op::Parameter>(element::i64, shape_b);
    Shape shape_c{3, 2};
    auto C = make_shared<op::Parameter>(element::i64, shape_c);
    Shape shape_r{8, 2};
    auto f = make_shared<Function>(make_shared<op::Concat>(NodeVector{A, B, C}, 0),
                                   ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i64, shape_a);
    copy_data(a, vector<int64_t>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::i64, shape_b);
    copy_data(b, vector<int64_t>{1, 2, 4, 8, 16, 32});
    auto c = backend->create_tensor(element::i64, shape_c);
    copy_data(c, vector<int64_t>{2, 3, 5, 7, 11, 13});
    auto result = backend->create_tensor(element::i64, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b, c});
    EXPECT_EQ((vector<int64_t>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13}),
              read_vector<int64_t>(result));
}

// Params to drive concat_vector_large testing variations
class concat_vector_params : public ::testing::TestWithParam<int>
{
protected:
    concat_vector_params() { num_inputs = GetParam(); }
    uint32_t num_inputs;
};

NGRAPH_TEST_P(${BACKEND_NAME}, concat_vector_params, concat_vector_large)
{
    Shape shape_a{1};
    NodeVector inputs;
    ParameterVector inputs_param;
    for (uint32_t i = 0; i < num_inputs; i++)
    {
        auto A = make_shared<op::Parameter>(element::f32, shape_a);
        inputs_param.push_back(A);
        inputs.push_back(A);
    }
    Shape shape_r{num_inputs};
    auto f = make_shared<Function>(make_shared<op::Concat>(inputs, 0), inputs_param);

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    std::vector<std::shared_ptr<runtime::Tensor>> inputs_value;
    std::vector<float> ref_result;
    for (uint32_t i = 0; i < num_inputs; i++)
    {
        auto a = backend->create_tensor(element::f32, shape_a);
        copy_data(a, vector<float>{static_cast<float>(i)});
        ref_result.push_back(static_cast<float>(i));
        inputs_value.push_back(a);
    }
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, inputs_value);
    EXPECT_TRUE(
        test::all_close_f(ref_result, read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

// concat_vector_large case generation
// Add thhosw tests to cover paramter space overflow:
// cuda kernel parameter space have limit, if there is large number of parameters,
// there will be overflow for parameter space.
NGRAPH_INSTANTIATE_TEST_SUITE_P(${BACKEND_NAME},
                                input_sizes,
                                concat_vector_params,
                                testing::Values(100, 128, 999));

NGRAPH_TEST(${BACKEND_NAME}, concat_vector)
{
    Shape shape_a{4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{6};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{2};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{12};
    auto f = make_shared<Function>(make_shared<op::Concat>(NodeVector{A, B, C}, 0),
                                   ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{1, 2, 4, 8, 16, 32});
    auto c = backend->create_tensor(element::f32, shape_c);
    copy_data(c, vector<float>{18, 19});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b, c});
    EXPECT_TRUE(test::all_close_f((vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 18, 19}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, concat_4d_tensor)
{
    Shape shape{1, 1, 1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    Shape shape_r{3, 1, 1, 1};
    auto f = make_shared<Function>(make_shared<op::Concat>(NodeVector{A, B, C}, 0),
                                   ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{2});
    auto c = backend->create_tensor(element::f32, shape);
    copy_data(c, vector<float>{3});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b, c});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1, 2, 3}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, concat_2d_tensor)
{
    Shape shape{1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    Shape shape_r{3, 1};
    auto f = make_shared<Function>(make_shared<op::Concat>(NodeVector{A, B, C}, 0),
                                   ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{2});
    auto c = backend->create_tensor(element::f32, shape);
    copy_data(c, vector<float>{3});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b, c});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1, 2, 3}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, concat_in_place_2d_tensor)
{
    Shape shape{1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto add1 = make_shared<op::v1::Add>(A, B);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto D = make_shared<op::Parameter>(element::f32, shape);
    auto add2 = make_shared<op::v1::Add>(C, D);
    auto subtract = make_shared<op::v1::Subtract>(C, A);
    Shape shape_r{3, 1};
    auto f = make_shared<Function>(make_shared<op::Concat>(NodeVector{add1, add2, subtract}, 0),
                                   ParameterVector{A, B, C, D});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{2});
    auto c = backend->create_tensor(element::f32, shape);
    copy_data(c, vector<float>{3});
    auto d = backend->create_tensor(element::f32, shape);
    copy_data(d, vector<float>{4});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b, c, d});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{3, 7, 2}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, concat_in_place_propagate_2d_tensor)
{
    Shape shape{1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto add1 = make_shared<op::v1::Add>(A, B);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto D = make_shared<op::Parameter>(element::f32, shape);
    auto add2 = make_shared<op::v1::Add>(C, D);
    auto concat1 = make_shared<op::Concat>(NodeVector{add1, add2}, 0);
    auto subtract = make_shared<op::v1::Subtract>(C, A);
    Shape shape_r{3, 1};
    auto f = make_shared<Function>(make_shared<op::Concat>(NodeVector{concat1, subtract}, 0),
                                   ParameterVector{A, B, C, D});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{2});
    auto c = backend->create_tensor(element::f32, shape);
    copy_data(c, vector<float>{3});
    auto d = backend->create_tensor(element::f32, shape);
    copy_data(d, vector<float>{4});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b, c, d});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{3, 7, 2}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, concat_in_place_tree_1)
{
    Shape shape{1, 2, 2};
    Shape shape_r{1, 4, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto add1 = make_shared<op::v1::Add>(A, B);
    auto add2 = make_shared<op::v1::Add>(A, B);
    auto concat = make_shared<op::Concat>(NodeVector{add1, add2}, 1);
    auto f = make_shared<Function>(make_shared<op::v1::Add>(concat, concat), ParameterVector{A, B});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 1, 1, 1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 1, 1, 1});

    auto result = backend->create_tensor(element::f32, shape_r);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    vector<float> expected;
    expected.resize(8, 4);

    EXPECT_TRUE(test::all_close_f(expected, read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, concat_in_place_tree_2)
{
    Shape shape{1, 2, 2};
    Shape shape_r{1, 8, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto add1 = make_shared<op::v1::Add>(A, B);
    auto add2 = make_shared<op::v1::Add>(A, B);
    auto concat1 = make_shared<op::Concat>(NodeVector{add1, add2}, 1);
    auto concat2 = make_shared<op::Concat>(NodeVector{add1, add2}, 1);
    auto concat12 = make_shared<op::Concat>(NodeVector{concat1, concat2}, 1);
    auto f =
        make_shared<Function>(make_shared<op::v1::Add>(concat12, concat12), ParameterVector{A, B});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 1, 1, 1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 1, 1, 1});
    auto result = backend->create_tensor(element::f32, shape_r);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    vector<float> expected;
    expected.resize(16, 4);

    EXPECT_TRUE(test::all_close_f(expected, read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, concat_in_place_tree_3)
{
    Shape shape{1, 2, 2};
    Shape shape_r{1, 16, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto concat1 = make_shared<op::Concat>(NodeVector{A, B}, 1);
    auto concat2 = make_shared<op::Concat>(NodeVector{A, B}, 1);
    auto concat3 = make_shared<op::Concat>(NodeVector{A, B}, 1);
    auto concat4 = make_shared<op::Concat>(NodeVector{A, B}, 1);
    auto concat12 = make_shared<op::Concat>(NodeVector{concat1, concat2}, 1);
    auto concat34 = make_shared<op::Concat>(NodeVector{concat3, concat4}, 1);
    auto concat14 = make_shared<op::Concat>(NodeVector{concat12, concat34}, 1);
    auto f =
        make_shared<Function>(make_shared<op::v1::Add>(concat14, concat14), ParameterVector{A, B});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 1, 1, 1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 1, 1, 1});
    auto result = backend->create_tensor(element::f32, shape_r);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    vector<float> expected;
    expected.resize(32, 2);

    EXPECT_TRUE(test::all_close_f(expected, read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, concat_in_place_add_concat)
{
    Shape shape{2, 2};
    Shape shape_r{4, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto add1 = make_shared<op::v1::Add>(A, B);
    auto add2 = make_shared<op::v1::Add>(add1, add1);
    auto concat = make_shared<op::Concat>(NodeVector{add1, add2}, 0);
    auto add3 = make_shared<op::v1::Add>(concat, concat);
    auto f = make_shared<Function>(add3, ParameterVector{A, B});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 1, 1, 1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 1, 1, 1});
    auto result = backend->create_tensor(element::f32, shape_r);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    vector<float> expected = {4, 4, 4, 4, 8, 8, 8, 8};
    EXPECT_TRUE(test::all_close_f(expected, read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, concat_in_place_add_concat_2)
{
    Shape shape{1, 2, 2};
    Shape shape_r{1, 6, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto add1 = make_shared<op::v1::Add>(A, B);
    auto add2 = make_shared<op::v1::Add>(A, B);
    auto add3 = make_shared<op::v1::Add>(A, B);
    auto add4 = make_shared<op::v1::Add>(A, B);
    auto add5 = make_shared<op::v1::Add>(A, B);

    auto concat1 = make_shared<op::Concat>(NodeVector{add1, add2, add3}, 1);

    auto concat2 = make_shared<op::Concat>(NodeVector{add4, add2, add5}, 1);

    auto add6 = make_shared<op::v1::Add>(concat1, concat2);
    auto f = make_shared<Function>(add6, ParameterVector{A, B});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 1, 1, 1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 1, 1, 1});
    auto result = backend->create_tensor(element::f32, shape_r);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    vector<float> expected = {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
    EXPECT_TRUE(test::all_close_f(expected, read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}
// from numpy import *
// a=linspace(1,2*3*4*3*2,2*3*4*3*2)
// b=linspace(1000+1,1000+2*3*3*3*2,2*3*3*3*2)
// c=linspace(2000+1,2000+2*3*2*3*2,2*3*2*3*2)
// a.shape=(2,3,4,3,2)
// b.shape=(2,3,3,3,2)
// c.shape=(2,3,2,3,2)
// z=concatenate((a,b,c),axis=2)
// z.shape=(2*3*(4+3+2)*3*2)
// set_printoptions(suppress=True)
// print(z)
//
// [    1.     2.     3.     4.     5.     6.     7.     8.     9.    10.
//     11.    12.    13.    14.    15.    16.    17.    18.    19.    20.
//     21.    22.    23.    24.  1001.  1002.  1003.  1004.  1005.  1006.
//   1007.  1008.  1009.  1010.  1011.  1012.  1013.  1014.  1015.  1016.
//   1017.  1018.  2001.  2002.  2003.  2004.  2005.  2006.  2007.  2008.
//   2009.  2010.  2011.  2012.    25.    26.    27.    28.    29.    30.
//     31.    32.    33.    34.    35.    36.    37.    38.    39.    40.
//     41.    42.    43.    44.    45.    46.    47.    48.  1019.  1020.
//   1021.  1022.  1023.  1024.  1025.  1026.  1027.  1028.  1029.  1030.
//   1031.  1032.  1033.  1034.  1035.  1036.  2013.  2014.  2015.  2016.
//   2017.  2018.  2019.  2020.  2021.  2022.  2023.  2024.    49.    50.
//     51.    52.    53.    54.    55.    56.    57.    58.    59.    60.
//     61.    62.    63.    64.    65.    66.    67.    68.    69.    70.
//     71.    72.  1037.  1038.  1039.  1040.  1041.  1042.  1043.  1044.
//   1045.  1046.  1047.  1048.  1049.  1050.  1051.  1052.  1053.  1054.
//   2025.  2026.  2027.  2028.  2029.  2030.  2031.  2032.  2033.  2034.
//   2035.  2036.    73.    74.    75.    76.    77.    78.    79.    80.
//     81.    82.    83.    84.    85.    86.    87.    88.    89.    90.
//     91.    92.    93.    94.    95.    96.  1055.  1056.  1057.  1058.
//   1059.  1060.  1061.  1062.  1063.  1064.  1065.  1066.  1067.  1068.
//   1069.  1070.  1071.  1072.  2037.  2038.  2039.  2040.  2041.  2042.
//   2043.  2044.  2045.  2046.  2047.  2048.    97.    98.    99.   100.
//    101.   102.   103.   104.   105.   106.   107.   108.   109.   110.
//    111.   112.   113.   114.   115.   116.   117.   118.   119.   120.
//   1073.  1074.  1075.  1076.  1077.  1078.  1079.  1080.  1081.  1082.
//   1083.  1084.  1085.  1086.  1087.  1088.  1089.  1090.  2049.  2050.
//   2051.  2052.  2053.  2054.  2055.  2056.  2057.  2058.  2059.  2060.
//    121.   122.   123.   124.   125.   126.   127.   128.   129.   130.
//    131.   132.   133.   134.   135.   136.   137.   138.   139.   140.
//    141.   142.   143.   144.  1091.  1092.  1093.  1094.  1095.  1096.
//   1097.  1098.  1099.  1100.  1101.  1102.  1103.  1104.  1105.  1106.
//   1107.  1108.  2061.  2062.  2063.  2064.  2065.  2066.  2067.  2068.
//   2069.  2070.  2071.  2072.]
NGRAPH_TEST(${BACKEND_NAME}, concat_5d)
{
    vector<float> a_data(2 * 3 * 4 * 3 * 2);
    for (int i = 0; i < 2 * 3 * 4 * 3 * 2; i++)
    {
        a_data[i] = float(i + 1);
    }

    vector<float> b_data(2 * 3 * 3 * 3 * 2);
    for (int i = 0; i < 2 * 3 * 3 * 3 * 2; i++)
    {
        b_data[i] = 1000 + float(i + 1);
    }

    vector<float> c_data(2 * 3 * 2 * 3 * 2);
    for (int i = 0; i < 2 * 3 * 2 * 3 * 2; i++)
    {
        c_data[i] = 2000 + float(i + 1);
    }

    Shape shape_a{2, 3, 4, 3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 3, 3, 3, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{2, 3, 2, 3, 2};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{2, 3, 9, 3, 2};

    auto r = make_shared<op::Concat>(NodeVector{A, B, C}, 2);
    auto f = make_shared<Function>(r, ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, b_data);
    auto c = backend->create_tensor(element::f32, shape_c);
    copy_data(c, c_data);

    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b, c});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{
            1.,    2.,    3.,    4.,    5.,    6.,    7.,    8.,    9.,    10.,   11.,   12.,
            13.,   14.,   15.,   16.,   17.,   18.,   19.,   20.,   21.,   22.,   23.,   24.,
            1001., 1002., 1003., 1004., 1005., 1006., 1007., 1008., 1009., 1010., 1011., 1012.,
            1013., 1014., 1015., 1016., 1017., 1018., 2001., 2002., 2003., 2004., 2005., 2006.,
            2007., 2008., 2009., 2010., 2011., 2012., 25.,   26.,   27.,   28.,   29.,   30.,
            31.,   32.,   33.,   34.,   35.,   36.,   37.,   38.,   39.,   40.,   41.,   42.,
            43.,   44.,   45.,   46.,   47.,   48.,   1019., 1020., 1021., 1022., 1023., 1024.,
            1025., 1026., 1027., 1028., 1029., 1030., 1031., 1032., 1033., 1034., 1035., 1036.,
            2013., 2014., 2015., 2016., 2017., 2018., 2019., 2020., 2021., 2022., 2023., 2024.,
            49.,   50.,   51.,   52.,   53.,   54.,   55.,   56.,   57.,   58.,   59.,   60.,
            61.,   62.,   63.,   64.,   65.,   66.,   67.,   68.,   69.,   70.,   71.,   72.,
            1037., 1038., 1039., 1040., 1041., 1042., 1043., 1044., 1045., 1046., 1047., 1048.,
            1049., 1050., 1051., 1052., 1053., 1054., 2025., 2026., 2027., 2028., 2029., 2030.,
            2031., 2032., 2033., 2034., 2035., 2036., 73.,   74.,   75.,   76.,   77.,   78.,
            79.,   80.,   81.,   82.,   83.,   84.,   85.,   86.,   87.,   88.,   89.,   90.,
            91.,   92.,   93.,   94.,   95.,   96.,   1055., 1056., 1057., 1058., 1059., 1060.,
            1061., 1062., 1063., 1064., 1065., 1066., 1067., 1068., 1069., 1070., 1071., 1072.,
            2037., 2038., 2039., 2040., 2041., 2042., 2043., 2044., 2045., 2046., 2047., 2048.,
            97.,   98.,   99.,   100.,  101.,  102.,  103.,  104.,  105.,  106.,  107.,  108.,
            109.,  110.,  111.,  112.,  113.,  114.,  115.,  116.,  117.,  118.,  119.,  120.,
            1073., 1074., 1075., 1076., 1077., 1078., 1079., 1080., 1081., 1082., 1083., 1084.,
            1085., 1086., 1087., 1088., 1089., 1090., 2049., 2050., 2051., 2052., 2053., 2054.,
            2055., 2056., 2057., 2058., 2059., 2060., 121.,  122.,  123.,  124.,  125.,  126.,
            127.,  128.,  129.,  130.,  131.,  132.,  133.,  134.,  135.,  136.,  137.,  138.,
            139.,  140.,  141.,  142.,  143.,  144.,  1091., 1092., 1093., 1094., 1095., 1096.,
            1097., 1098., 1099., 1100., 1101., 1102., 1103., 1104., 1105., 1106., 1107., 1108.,
            2061., 2062., 2063., 2064., 2065., 2066., 2067., 2068., 2069., 2070., 2071., 2072.}),
        read_vector<float>(result),
        MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, concat_zero_length_1d_last)
{
    Shape shape_a{4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{0};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{4};

    auto r = make_shared<op::Concat>(NodeVector{A, B}, 0);
    auto f = make_shared<Function>(r, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    vector<float> a_data{1, 2, 3, 4};
    vector<float> b_data(0);

    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, b_data);

    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1, 2, 3, 4}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, concat_zero_length_1d_middle)
{
    Shape shape_a{4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{0};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{4};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{8};

    auto r = make_shared<op::Concat>(NodeVector{A, B, C}, 0);
    auto f = make_shared<Function>(r, ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    vector<float> a_data{1, 2, 3, 4};
    vector<float> b_data(0);
    vector<float> c_data{5, 6, 7, 8};

    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, b_data);
    auto c = backend->create_tensor(element::f32, shape_c);
    copy_data(c, c_data);

    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b, c});
    EXPECT_TRUE(test::all_close_f((vector<float>{1, 2, 3, 4, 5, 6, 7, 8}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, concat_zero_zero)
{
    Shape shape{0};
    auto constant_1 = op::Constant::create(element::f32, shape, {1});
    auto concat_1 = make_shared<op::Concat>(NodeVector{constant_1, constant_1}, 0);

    auto f = make_shared<Function>(concat_1, ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {});

    EXPECT_TRUE(
        test::all_close_f(vector<float>{}, read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, concat_zero_length_4d_middle)
{
    Shape shape_a{2, 2, 1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 2, 0, 1};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{2, 2, 1, 1};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{2, 2, 2, 1};

    auto r = make_shared<op::Concat>(NodeVector{A, B, C}, 2);
    auto f = make_shared<Function>(r, ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    vector<float> a_data{1, 2, 3, 4};
    vector<float> b_data(0);
    vector<float> c_data{5, 6, 7, 8};

    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, b_data);
    auto c = backend->create_tensor(element::f32, shape_c);
    copy_data(c, c_data);

    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b, c});
    EXPECT_TRUE(test::all_close_f((vector<float>{1, 5, 2, 6, 3, 7, 4, 8}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}
