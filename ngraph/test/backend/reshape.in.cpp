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
#include <numeric>
#include <random>
#include <string>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "runtime/backend.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/engine/test_engines.hpp"
#include "util/ndarray.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, reshape_t2v_012)
{
    Shape shape_a{2, 2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{12};
    auto r = make_shared<op::Reshape>(A, AxisVector{0, 1, 2}, shape_r);
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_t2s_012)
{
    Shape shape_a{1, 1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{};
    auto r = make_shared<op::Reshape>(A, AxisVector{0, 1, 2}, shape_r);
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{6});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{6}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_t2s_120)
{
    Shape shape_a{1, 1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{};
    auto r = make_shared<op::Reshape>(A, AxisVector{1, 2, 0}, shape_r);
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{6});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{6}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_s2t)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 1, 1, 1, 1, 1};
    auto r = make_shared<op::Reshape>(A, AxisVector{}, shape_r);
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{42});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{42}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_s2t1)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::boolean, shape_a);
    Shape shape_r{1};
    auto r = make_shared<op::Reshape>(A, AxisVector{}, shape_r);
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::boolean, shape_a);
    copy_data(a, vector<char>{42});
    auto result = backend->create_tensor(element::boolean, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<char>{42}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_v2m_col)
{
    Shape shape_a{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3, 1};
    auto r = make_shared<op::Reshape>(A, AxisVector{0}, shape_r);
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1, 2, 3}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_v2m_row)
{
    Shape shape_a{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 3};
    auto r = make_shared<op::Reshape>(A, AxisVector{0}, shape_r);
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1, 2, 3}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_v2t_middle)
{
    Shape shape_a{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 3, 1};
    auto r = make_shared<op::Reshape>(A, AxisVector{0}, shape_r);
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1, 2, 3}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_m2m_same)
{
    Shape shape_a{3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3, 3};
    auto r = make_shared<op::Reshape>(A, AxisVector{0, 1}, shape_r);
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_m2m_transpose)
{
    Shape shape_a{3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3, 3};
    auto r = make_shared<op::Reshape>(A, AxisVector{1, 0}, shape_r);
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{1, 4, 7, 2, 5, 8, 3, 6, 9}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_m2m_dim_change_transpose)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 3};
    auto r = make_shared<op::Reshape>(A, AxisVector{1, 0}, shape_r);
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1, 3, 5, 2, 4, 6}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_3d_transpose_021)
{
    Shape shape_a{2, 3, 4};
    Shape shape_r{2, 4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto r = make_shared<op::Reshape>(A, AxisVector{0, 2, 1}, shape_r);
    auto f = make_shared<Function>(r, ParameterVector{A});

    vector<float> a_data(shape_size(shape_a));
    iota(a_data.begin(), a_data.end(), 1.f);

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, a_data);
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{1,  5,  9,  2,  6,  10, 3,  7,  11, 4,  8,  12,
                                                 13, 17, 21, 14, 18, 22, 15, 19, 23, 16, 20, 24}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_3d_transpose_210)
{
    Shape shape_a{2, 3, 4};
    Shape shape_r{4, 3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto r = make_shared<op::Reshape>(A, AxisVector{2, 1, 0}, shape_r);
    auto f = make_shared<Function>(r, ParameterVector{A});

    vector<float> a_data(shape_size(shape_a));
    iota(a_data.begin(), a_data.end(), 1.f);

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, a_data);
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{1, 13, 5, 17, 9,  21, 2, 14, 6, 18, 10, 22,
                                                 3, 15, 7, 19, 11, 23, 4, 16, 8, 20, 12, 24}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_3d_transpose_201)
{
    Shape shape_a{2, 3, 4};
    Shape shape_r{4, 2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto r = make_shared<op::Reshape>(A, AxisVector{2, 0, 1}, shape_r);
    auto f = make_shared<Function>(r, ParameterVector{A});

    vector<float> a_data(shape_size(shape_a));
    iota(a_data.begin(), a_data.end(), 1.f);

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, a_data);
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{1, 5, 9,  13, 17, 21, 2, 6, 10, 14, 18, 22,
                                                 3, 7, 11, 15, 19, 23, 4, 8, 12, 16, 20, 24}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_3d_transpose_102)
{
    Shape shape_a{2, 3, 4};
    Shape shape_r{3, 2, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto r = make_shared<op::Reshape>(A, AxisVector{1, 0, 2}, shape_r);
    auto f = make_shared<Function>(r, ParameterVector{A});

    vector<float> a_data(shape_size(shape_a));
    iota(a_data.begin(), a_data.end(), 1.f);

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, a_data);
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{1,  2,  3,  4,  13, 14, 15, 16, 5,  6,  7,  8,
                                                 17, 18, 19, 20, 9,  10, 11, 12, 21, 22, 23, 24}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_3d_transpose_120)
{
    Shape shape_a{2, 3, 4};
    Shape shape_r{3, 4, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto r = make_shared<op::Reshape>(A, AxisVector{1, 2, 0}, shape_r);
    auto f = make_shared<Function>(r, ParameterVector{A});

    vector<float> a_data(shape_size(shape_a));
    iota(a_data.begin(), a_data.end(), 1.f);

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, a_data);
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{1, 13, 2, 14, 3, 15, 4,  16, 5,  17, 6,  18,
                                                 7, 19, 8, 20, 9, 21, 10, 22, 11, 23, 12, 24}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_4d_transpose)
{
    Shape shape_a{2, 2, 5, 5};
    Shape shape_r{2, 5, 5, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto r = make_shared<op::Reshape>(A, AxisVector{0, 2, 3, 1}, shape_r);
    auto f = make_shared<Function>(r, ParameterVector{A});

    vector<float> a_data(shape_size(shape_a));
    iota(a_data.begin(), a_data.end(), 1.f);

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, a_data);
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1.,  26., 2.,  27., 3.,  28., 4.,  29., 5.,  30., 6.,  31., 7.,  32., 8.,
                       33., 9.,  34., 10., 35., 11., 36., 12., 37., 13., 38., 14., 39., 15., 40.,
                       16., 41., 17., 42., 18., 43., 19., 44., 20., 45., 21., 46., 22., 47., 23.,
                       48., 24., 49., 25., 50., 51., 76., 52., 77., 53., 78., 54., 79., 55., 80.,
                       56., 81., 57., 82., 58., 83., 59., 84., 60., 85., 61., 86., 62., 87., 63.,
                       88., 64., 89., 65., 90., 66., 91., 67., 92., 68., 93., 69., 94., 70., 95.,
                       71., 96., 72., 97., 73., 98., 74., 99., 75., 100.}),
        read_vector<float>(result),
        MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_4d_no_transpose)
{
    Shape shape_a{2, 2, 5, 5};
    Shape shape_r{2, 5, 5, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto r = make_shared<op::Reshape>(A, AxisVector{0, 1, 2, 3}, shape_r);
    auto f = make_shared<Function>(r, ParameterVector{A});

    vector<float> a_data(shape_size(shape_a));
    iota(a_data.begin(), a_data.end(), 1.f);

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, a_data);
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(a_data, read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_transposed_shape_change)
{
    Shape shape_a{2, 6};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{12};
    auto r = make_shared<op::Reshape>(A, AxisVector{1, 0}, shape_r);
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6, 12}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

//
// Numpy:
//
// >>> x = linspace(1,2*2*3*3*2*4,2*2*3*3*2*4)
// >>> x.shape=(2,2,3,3,2,4)
// >>> y = ascontiguousarray(transpose(x,(2,4,0,5,3,1)))
// >>> y.shape=2*2*3*3*2*4
// >>> y
// array([   1.,   73.,    9.,   81.,   17.,   89.,    2.,   74.,   10.,
//          82.,   18.,   90.,    3.,   75.,   11.,   83.,   19.,   91.,
//           4.,   76.,   12.,   84.,   20.,   92.,  145.,  217.,  153.,
//         225.,  161.,  233.,  146.,  218.,  154.,  226.,  162.,  234.,
//         147.,  219.,  155.,  227.,  163.,  235.,  148.,  220.,  156.,
//         228.,  164.,  236.,    5.,   77.,   13.,   85.,   21.,   93.,
//           6.,   78.,   14.,   86.,   22.,   94.,    7.,   79.,   15.,
//          87.,   23.,   95.,    8.,   80.,   16.,   88.,   24.,   96.,
//         149.,  221.,  157.,  229.,  165.,  237.,  150.,  222.,  158.,
//         230.,  166.,  238.,  151.,  223.,  159.,  231.,  167.,  239.,
//         152.,  224.,  160.,  232.,  168.,  240.,   25.,   97.,   33.,
//         105.,   41.,  113.,   26.,   98.,   34.,  106.,   42.,  114.,
//          27.,   99.,   35.,  107.,   43.,  115.,   28.,  100.,   36.,
//         108.,   44.,  116.,  169.,  241.,  177.,  249.,  185.,  257.,
//         170.,  242.,  178.,  250.,  186.,  258.,  171.,  243.,  179.,
//         251.,  187.,  259.,  172.,  244.,  180.,  252.,  188.,  260.,
//          29.,  101.,   37.,  109.,   45.,  117.,   30.,  102.,   38.,
//         110.,   46.,  118.,   31.,  103.,   39.,  111.,   47.,  119.,
//          32.,  104.,   40.,  112.,   48.,  120.,  173.,  245.,  181.,
//         253.,  189.,  261.,  174.,  246.,  182.,  254.,  190.,  262.,
//         175.,  247.,  183.,  255.,  191.,  263.,  176.,  248.,  184.,
//         256.,  192.,  264.,   49.,  121.,   57.,  129.,   65.,  137.,
//          50.,  122.,   58.,  130.,   66.,  138.,   51.,  123.,   59.,
//         131.,   67.,  139.,   52.,  124.,   60.,  132.,   68.,  140.,
//         193.,  265.,  201.,  273.,  209.,  281.,  194.,  266.,  202.,
//         274.,  210.,  282.,  195.,  267.,  203.,  275.,  211.,  283.,
//         196.,  268.,  204.,  276.,  212.,  284.,   53.,  125.,   61.,
//         133.,   69.,  141.,   54.,  126.,   62.,  134.,   70.,  142.,
//          55.,  127.,   63.,  135.,   71.,  143.,   56.,  128.,   64.,
//         136.,   72.,  144.,  197.,  269.,  205.,  277.,  213.,  285.,
//         198.,  270.,  206.,  278.,  214.,  286.,  199.,  271.,  207.,
//         279.,  215.,  287.,  200.,  272.,  208.,  280.,  216.,  288.])
//
NGRAPH_TEST(${BACKEND_NAME}, reshape_6d)
{
    Shape shape_a{2, 2, 3, 3, 2, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3, 2, 2, 4, 3, 2};

    vector<float> a_data(shape_size(shape_a));
    iota(a_data.begin(), a_data.end(), 1.f);

    auto r = make_shared<op::Reshape>(A, AxisVector{2, 4, 0, 5, 3, 1}, shape_r);
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, a_data);

    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{
            1.,   73.,  9.,   81.,  17.,  89.,  2.,   74.,  10.,  82.,  18.,  90.,  3.,   75.,
            11.,  83.,  19.,  91.,  4.,   76.,  12.,  84.,  20.,  92.,  145., 217., 153., 225.,
            161., 233., 146., 218., 154., 226., 162., 234., 147., 219., 155., 227., 163., 235.,
            148., 220., 156., 228., 164., 236., 5.,   77.,  13.,  85.,  21.,  93.,  6.,   78.,
            14.,  86.,  22.,  94.,  7.,   79.,  15.,  87.,  23.,  95.,  8.,   80.,  16.,  88.,
            24.,  96.,  149., 221., 157., 229., 165., 237., 150., 222., 158., 230., 166., 238.,
            151., 223., 159., 231., 167., 239., 152., 224., 160., 232., 168., 240., 25.,  97.,
            33.,  105., 41.,  113., 26.,  98.,  34.,  106., 42.,  114., 27.,  99.,  35.,  107.,
            43.,  115., 28.,  100., 36.,  108., 44.,  116., 169., 241., 177., 249., 185., 257.,
            170., 242., 178., 250., 186., 258., 171., 243., 179., 251., 187., 259., 172., 244.,
            180., 252., 188., 260., 29.,  101., 37.,  109., 45.,  117., 30.,  102., 38.,  110.,
            46.,  118., 31.,  103., 39.,  111., 47.,  119., 32.,  104., 40.,  112., 48.,  120.,
            173., 245., 181., 253., 189., 261., 174., 246., 182., 254., 190., 262., 175., 247.,
            183., 255., 191., 263., 176., 248., 184., 256., 192., 264., 49.,  121., 57.,  129.,
            65.,  137., 50.,  122., 58.,  130., 66.,  138., 51.,  123., 59.,  131., 67.,  139.,
            52.,  124., 60.,  132., 68.,  140., 193., 265., 201., 273., 209., 281., 194., 266.,
            202., 274., 210., 282., 195., 267., 203., 275., 211., 283., 196., 268., 204., 276.,
            212., 284., 53.,  125., 61.,  133., 69.,  141., 54.,  126., 62.,  134., 70.,  142.,
            55.,  127., 63.,  135., 71.,  143., 56.,  128., 64.,  136., 72.,  144., 197., 269.,
            205., 277., 213., 285., 198., 270., 206., 278., 214., 286., 199., 271., 207., 279.,
            215., 287., 200., 272., 208., 280., 216., 288.}),
        read_vector<float>(result),
        MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, builder_reshape_1D_to_scalar)
{
    const Shape input_shape{1};
    const auto input = make_shared<op::Parameter>(element::f32, input_shape);
    const auto reshape_builder = builder::opset1::reshape(input, Shape{});
    auto function = make_shared<Function>(reshape_builder, ParameterVector{input});

    auto test_case = test::TestCase<TestEngine>(function);
    vector<float> input_values(shape_size(input_shape), 1.f);
    test_case.add_input<float>(input_shape, input_values);
    test_case.add_expected_output<float>(Shape{}, vector<float>{1.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, builder_reshape_3D_to_scalar)
{
    const Shape input_shape{1, 1, 1};
    const auto input = make_shared<op::Parameter>(element::f32, input_shape);
    const auto reshape_builder = builder::opset1::reshape(input, Shape{});
    auto function = make_shared<Function>(reshape_builder, ParameterVector{input});

    auto test_case = test::TestCase<TestEngine>(function);
    vector<float> input_values(shape_size(input_shape), 1.f);
    test_case.add_input<float>(input_shape, input_values);
    test_case.add_expected_output<float>(Shape{}, vector<float>{1.f});

    test_case.run();
}

#if NGRAPH_INTERPRETER_ENABLE

NGRAPH_TEST(${BACKEND_NAME}, reshape_shufflenet_5d)
{
    Shape shape_a{1, 112, 56, 56};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{1, 4, 28, 56, 56};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{1, 28, 4, 56, 56};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{1, 112, 56, 56};

    vector<float> a_data(shape_size(shape_a));
    iota(a_data.begin(), a_data.end(), 1.f);

    auto r0 = make_shared<op::Reshape>(A, AxisVector{0, 1, 2, 3}, shape_b);
    auto r1 = make_shared<op::Reshape>(r0, AxisVector{0, 2, 1, 3, 4}, shape_c);
    auto r2 = make_shared<op::Reshape>(r1, AxisVector{0, 1, 2, 3, 4}, shape_r);
    auto f = make_shared<Function>(r2, ParameterVector{A});

    auto ref_func = clone_function(*f);
    auto bk_func = clone_function(*f);

    vector<vector<float>> args;
    args.push_back(a_data);

    auto ref_results = execute(ref_func, args, "INTERPRETER");
    auto bk_results = execute(bk_func, args, "${BACKEND_NAME}");

    EXPECT_TRUE(test::all_close_f(ref_results.at(0), bk_results.at(0), MIN_FLOAT_TOLERANCE_BITS));
}

#endif // NGRAPH_INTERPRETER_ENABLE
