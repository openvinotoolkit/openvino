//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>

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

//
// Numpy test:
//
// from numpy import *
// x = linspace(1,2*3*3*4,2*3*3*4)
// y = linspace(1,3*4*2*3*2,3*4*2*2*3)
// x.shape=(2,3,3,4)
// y.shape=(3,4,2,2,3)
// z = tensordot(x,y,([2,3],[0,1]))
// z.shape = 2*3*2*2*3
// z
//
// array([  6942.,   7020.,   7098.,   7176.,   7254.,   7332.,   7410.,
//          7488.,   7566.,   7644.,   7722.,   7800.,  16590.,  16812.,
//         17034.,  17256.,  17478.,  17700.,  17922.,  18144.,  18366.,
//         18588.,  18810.,  19032.,  26238.,  26604.,  26970.,  27336.,
//         27702.,  28068.,  28434.,  28800.,  29166.,  29532.,  29898.,
//         30264.,  35886.,  36396.,  36906.,  37416.,  37926.,  38436.,
//         38946.,  39456.,  39966.,  40476.,  40986.,  41496.,  45534.,
//         46188.,  46842.,  47496.,  48150.,  48804.,  49458.,  50112.,
//         50766.,  51420.,  52074.,  52728.,  55182.,  55980.,  56778.,
//         57576.,  58374.,  59172.,  59970.,  60768.,  61566.,  62364.,
//         63162.,  63960.])
//
NGRAPH_TEST(${BACKEND_NAME}, dot_4d_5d_multi_axis)
{
    vector<float> a_data(2 * 3 * 3 * 4);
    for (int i = 0; i < 2 * 3 * 3 * 4; i++)
    {
        a_data[i] = float(i + 1);
    }

    vector<float> b_data(3 * 4 * 2 * 2 * 3);
    for (int i = 0; i < 3 * 4 * 2 * 2 * 3; i++)
    {
        b_data[i] = float(i + 1);
    }

    Shape shape_a{2, 3, 3, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{3, 4, 2, 3, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{2, 3, 2, 3, 2};

    auto r = make_shared<op::Dot>(A, B, 2);
    auto f = make_shared<Function>(r, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, b_data);

    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{6942.,  7020.,  7098.,  7176.,  7254.,  7332.,  7410.,  7488.,  7566.,
                       7644.,  7722.,  7800.,  16590., 16812., 17034., 17256., 17478., 17700.,
                       17922., 18144., 18366., 18588., 18810., 19032., 26238., 26604., 26970.,
                       27336., 27702., 28068., 28434., 28800., 29166., 29532., 29898., 30264.,
                       35886., 36396., 36906., 37416., 37926., 38436., 38946., 39456., 39966.,
                       40476., 40986., 41496., 45534., 46188., 46842., 47496., 48150., 48804.,
                       49458., 50112., 50766., 51420., 52074., 52728., 55182., 55980., 56778.,
                       57576., 58374., 59172., 59970., 60768., 61566., 62364., 63162., 63960.}),
        read_vector<float>(result)));
}

//
// Numpy test:
//
// from numpy import *
// x = linspace(1,2*3*3*4,2*3*3*4)
// y = linspace(1,2*3*3*4*2,2*3*3*4*2)
// x.shape=(2,3,3,4)
// y.shape=(2,3,3,4,2)
// z = tensordot(x,y,([0,1,2,3],[0,1,2,3]))
// z
//
// array([ 251412.,  254040.])
//
NGRAPH_TEST(${BACKEND_NAME}, dot_4d_5d_multi_axis_more)
{
    vector<float> a_data(2 * 3 * 3 * 4);
    for (int i = 0; i < 2 * 3 * 3 * 4; i++)
    {
        a_data[i] = float(i + 1);
    }

    vector<float> b_data(2 * 3 * 3 * 4 * 2);
    for (int i = 0; i < 2 * 3 * 3 * 4 * 2; i++)
    {
        b_data[i] = float(i + 1);
    }

    Shape shape_a{2, 3, 3, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 3, 3, 4, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{2};

    auto r = make_shared<op::Dot>(A, B, 4);
    auto f = make_shared<Function>(r, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, b_data);

    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((vector<float>{251412., 254040.}), read_vector<float>(result)));
}

//
// Numpy test:
//
// from numpy import *
// x = linspace(1,20*30*30*40,20*30*30*40)
// y = linspace(1,20*30*30*40*20,20*30*30*40*20)
// x.shape=(20,30,30,40)
// y.shape=(20,30,30,40,20)
// z = tensordot(x,y,([0,1,2,3],[0,1,2,3]))
// set_printoptions(precision=20)
// z
//
// array([  2.48832025919525478400e+18,   2.48832051839533977600e+18,
//          2.48832077759658444800e+18,   2.48832103679413504000e+18,
//          2.48832129599669350400e+18,   2.48832155519793971200e+18,
//          2.48832181439802265600e+18,   2.48832207359808000000e+18,
//          2.48832233279813580800e+18,   2.48832259199822028800e+18,
//          2.48832285119946496000e+18,   2.48832311040043008000e+18,
//          2.48832336959957401600e+18,   2.48832362880081817600e+18,
//          2.48832388800090368000e+18,   2.48832414720096000000e+18,
//          2.48832440640101478400e+18,   2.48832466560109772800e+18,
//          2.48832492480234188800e+18,   2.48832518400031897600e+18])
//
// Disabled because this test is very slow.
//
NGRAPH_TEST(DISABLED_${BACKEND_NAME}, dot_4d_5d_multi_axis_big_fp64_VERY_SLOW)
{
    vector<double> a_data(20 * 30 * 30 * 40);
    for (int i = 0; i < 20 * 30 * 30 * 40; i++)
    {
        a_data[i] = double(i + 1);
    }

    vector<double> b_data(20 * 30 * 30 * 40 * 20);
    for (int i = 0; i < 20 * 30 * 30 * 40 * 20; i++)
    {
        b_data[i] = double(i + 1);
    }

    Shape shape_a{20, 30, 30, 40};
    auto A = make_shared<op::Parameter>(element::f64, shape_a);
    Shape shape_b{20, 30, 30, 40, 20};
    auto B = make_shared<op::Parameter>(element::f64, shape_b);
    Shape shape_r{20};

    auto r = make_shared<op::Dot>(A, B, 4);
    auto f = make_shared<Function>(r, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f64, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::f64, shape_b);
    copy_data(b, b_data);

    auto result = backend->create_tensor(element::f64, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f(
        vector<double>{
            2.48832025919525478400e+18, 2.48832051839533977600e+18, 2.48832077759658444800e+18,
            2.48832103679413504000e+18, 2.48832129599669350400e+18, 2.48832155519793971200e+18,
            2.48832181439802265600e+18, 2.48832207359808000000e+18, 2.48832233279813580800e+18,
            2.48832259199822028800e+18, 2.48832285119946496000e+18, 2.48832311040043008000e+18,
            2.48832336959957401600e+18, 2.48832362880081817600e+18, 2.48832388800090368000e+18,
            2.48832414720096000000e+18, 2.48832440640101478400e+18, 2.48832466560109772800e+18,
            2.48832492480234188800e+18, 2.48832518400031897600e+18},
        read_vector<double>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, dot_0_0)
{
    Shape shape{0};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    Shape shape_r{};
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_r);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the
    // right value.
    copy_data(result, vector<float>{2112});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((vector<float>{0}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, dot_matrix_2x0_0x2)
{
    Shape shape_a{2, 0};
    Shape shape_b{0, 2};
    Shape shape_r{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_r);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the
    // right value.
    copy_data(result, vector<float>{2112, 2112, 2112, 2112});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((vector<float>{0, 0, 0, 0}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, dot_matrix_0x2_2x0)
{
    Shape shape_a{0, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 0};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{0, 0};
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((vector<float>{}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, dot_matrix_3x2_2x0)
{
    Shape shape_a{3, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 0};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{3, 0};
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((vector<float>{}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, dot_scalar_0x2)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{0, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{0, 2};
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((vector<float>{}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, dot_2x0_0)
{
    Shape shape_a{2, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{0};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{2};
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_r);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the
    // right value.
    copy_data(result, vector<float>{2112, 2112});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((vector<float>{0, 0}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, dot1d)
{
    Shape shape{4};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    Shape shape_r{};
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((vector<float>{170}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, dot2d)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    Shape shape_r{2, 2};
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{5, 6, 7, 8});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((vector<float>{19, 22, 43, 50}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, dot2d_non_square)
{
    Shape shape_in1{2, 3};
    Shape shape_in2{3, 3};
    Shape shape_out{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_in1);
    auto B = make_shared<op::Parameter>(element::f32, shape_in2);
    auto dot = make_shared<op::Dot>(A, B);
    auto f = make_shared<Function>(dot, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape_in1);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape_in2);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape_out);

    copy_data(a, vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    copy_data(b, vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f(read_vector<float>(result),
                                  vector<float>{30.f, 36.f, 42.f, 66.f, 81.f, 96.f}));
}

//
// Here is what numpy does:
//
// >>> a = linspace(1,2*2*2,2*2*2)
// >>> b = linspace(1,2*2*2,2*2*2)
//
// >>> a.shape=(2,2,2)
// >>> b.shape=(2,2,2)
//
// >>> tensordot(a,b,axes=([2],[0]))
// array([[[[ 11.,  14.],
//          [ 17.,  20.]],
//
//         [[ 23.,  30.],
//          [ 37.,  44.]]],
//
//
//        [[[ 35.,  46.],
//          [ 57.,  68.]],
//
//         [[ 47.,  62.],
//          [ 77.,  92.]]]])
//
NGRAPH_TEST(${BACKEND_NAME}, dot3d_3d)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    Shape shape_r{2, 2, 2, 2};
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{11, 14, 17, 20, 23, 30, 37, 44, 35, 46, 57, 68, 47, 62, 77, 92}),
        read_vector<float>(result)));
}

//
// Here is what numpy does:
//
// >>> from numpy import *
// >>> a = linspace(0,4*2*3-1,4*2*3)
// >>> b = linspace(0,3*4-1,3*4)
//
// >>> a.shape=(4,2,3)
// >>> b.shape=(3,4)
//
// >>> tensordot(a,b,axes=([2],[0]))
// array([[[  20.,   23.,   26.,   29.],
//         [  56.,   68.,   80.,   92.]],
//
//        [[  92.,  113.,  134.,  155.],
//         [ 128.,  158.,  188.,  218.]],
//
//        [[ 164.,  203.,  242.,  281.],
//         [ 200.,  248.,  296.,  344.]],
//
//        [[ 236.,  293.,  350.,  407.],
//         [ 272.,  338.,  404.,  470.]]])
//
NGRAPH_TEST(${BACKEND_NAME}, dot3d_2d)
{
    Shape shape_a{4, 2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{3, 4};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{4, 2, 4};
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                               12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(
        test::all_close_f((vector<float>{20,  23,  26,  29,  56,  68,  80,  92,  92,  113, 134,
                                         155, 128, 158, 188, 218, 164, 203, 242, 281, 200, 248,
                                         296, 344, 236, 293, 350, 407, 272, 338, 404, 470}),
                          read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, dot_scalar_tensor_arg0)
{
    Shape shape_a{};
    Shape shape_b{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{6});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    auto result = backend->create_tensor(element::f32, shape_b);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((vector<float>{6, 12, 18, 24, 30, 36, 42, 48}),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, dot_scalar_tensor_arg1)
{
    Shape shape_a{2, 2, 2};
    Shape shape_b{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{6});
    auto result = backend->create_tensor(element::f32, shape_a);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((vector<float>{6, 12, 18, 24, 30, 36, 42, 48}),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, dot_scalar_scalar)
{
    Shape shape{};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{8});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{6});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((vector<float>{48}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, dot_matrix_vector_4_3)
{
    Shape shape_a{4, 3};
    Shape shape_b{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), ParameterVector{A, B});
    Shape shape_r{4};

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{17, 18, 19});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((vector<float>{110, 272, 434, 596}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, dot_matrix_vector)
{
    Shape shape_a{4, 4};
    Shape shape_b{4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), ParameterVector{A, B});
    Shape shape_r{4};

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{17, 18, 19, 20});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(
        test::all_close_f((vector<float>{190, 486, 782, 1078}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, dot_matrix_vector_int64)
{
    Shape shape_a{4, 4};
    Shape shape_b{4};
    auto A = make_shared<op::Parameter>(element::i64, shape_a);
    auto B = make_shared<op::Parameter>(element::i64, shape_b);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), ParameterVector{A, B});
    Shape shape_r{4};

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i64, shape_a);
    copy_data(a, vector<int64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto b = backend->create_tensor(element::i64, shape_b);
    copy_data(b, vector<int64_t>{17, 18, 19, 20});
    auto result = backend->create_tensor(element::i64, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ((vector<int64_t>{190, 486, 782, 1078}), read_vector<int64_t>(result));
}

//
// Numpy test:
//
// > from numpy import *
// > x = linspace(1,2*3*4,2*3*4)
// > y = linspace(1,3*4*5,3*4*5)
// > x.shape=(2,3,4)
// > y.shape=(3,4,5)
// > z = tensordot(x,y,([1,2],[0,1]))
// > z.shape = 2*5
// > z
// array([ 2938.,  3016.,  3094.,  3172.,  3250.,  7042.,  7264.,  7486.,
//         7708.,  7930.])
//
NGRAPH_TEST(${BACKEND_NAME}, dot_3d_multi_axis)
{
    vector<float> a_data(2 * 3 * 4);
    for (int i = 0; i < 2 * 3 * 4; i++)
    {
        a_data[i] = float(i + 1);
    }

    vector<float> b_data(3 * 4 * 5);
    for (int i = 0; i < 3 * 4 * 5; i++)
    {
        b_data[i] = float(i + 1);
    }

    Shape shape_a{2, 3, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{3, 4, 5};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{2, 5};

    auto r = make_shared<op::Dot>(A, B, 2);
    auto f = make_shared<Function>(r, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, b_data);

    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{2938., 3016., 3094., 3172., 3250., 7042., 7264., 7486., 7708., 7930.}),
        read_vector<float>(result)));
}

//
// Numpy test:
//
// > from numpy import *
// > x = array([6,61,2,3,5,21,75,23,23,0,23,2,35,67,1,2,9,16,2,3,6,1,8,0])
// > y = array([9,1,4,6,3,5,1,36,7,3,5,0,1,20,35,2,1,0,1,25,3,6,7,8])
// > x.shape=(2,4,3)
// > y.shape=(3,4,2)
// > z = tensordot(x,y,([2],[0]))
// > z.shape = 2*4*4*2
// > z
// array([ 483,  189,  331,   86,   85, 1262, 2155,  354,   83,   18,   58,
//         543,   77,  241,  325,  286,  859,  144,  438, 1025,  317,  973,
//        1041, 2930,  163,   69,  117,   50,   29,  472,  819,   62,  785,
//         236,  476,  235,  175, 1521, 2387, 1402,   97,   29,   69,  412,
//          63,  286,  429,  218,   45,   11,   29,  162,   27,  106,  149,
//         126,   65,   25,   44,    6,   11,  165,  281,   52])
//
NGRAPH_TEST(${BACKEND_NAME}, dot_3d_one_axis_arbitrary)
{
    vector<float> a_data{6,  61, 2, 3, 5, 21, 75, 23, 23, 0, 23, 2,
                         35, 67, 1, 2, 9, 16, 2,  3,  6,  1, 8,  0};
    vector<float> b_data{9, 1,  4,  6, 3, 5, 1, 36, 7, 3, 5, 0,
                         1, 20, 35, 2, 1, 0, 1, 25, 3, 6, 7, 8};

    Shape shape_a{2, 4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{3, 4, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{2, 4, 4, 2};

    auto r = make_shared<op::Dot>(A, B);
    auto f = make_shared<Function>(r, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, b_data);

    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{483,  189, 331, 86,  85,  1262, 2155, 354, 83,  18,   58,   543,  77,
                       241,  325, 286, 859, 144, 438,  1025, 317, 973, 1041, 2930, 163,  69,
                       117,  50,  29,  472, 819, 62,   785,  236, 476, 235,  175,  1521, 2387,
                       1402, 97,  29,  69,  412, 63,   286,  429, 218, 45,   11,   29,   162,
                       27,   106, 149, 126, 65,  25,   44,   6,   11,  165,  281,  52}),
        read_vector<float>(result)));
}
