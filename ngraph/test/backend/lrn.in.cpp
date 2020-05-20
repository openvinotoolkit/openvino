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

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif

#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on

#include <numeric>

#include "gtest/gtest.h"
#include "runtime/backend.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, lrn_across_channel)
{
    Shape shape{2, 3, 2, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    double alpha = 3;
    double beta = 0.5;
    double bias = 1;
    size_t size = 3;
    // lrn is performed across channel as default
    auto lrn = make_shared<op::LRN>(A, alpha, beta, bias, size);
    auto f = make_shared<Function>(lrn, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    vector<float> args{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, args);

    auto result = backend->create_tensor(element::f32, shape);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});

    vector<float> expected{0.f,
                           0.3015113f,
                           0.4364357f,
                           0.5f,
                           0.8728715f,
                           0.8451542f,
                           0.5970223f,
                           0.6115928f,
                           0.5642765f,
                           0.5669467f,
                           0.7784989f,
                           0.7720487f};
    EXPECT_TRUE(test::all_close_f(expected, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, lrn_across_h)
{
    Shape shape{2, 3, 2, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axes = make_shared<op::Constant>(element::i64, Shape{1}, vector<int64_t>{2});
    double alpha = 3;
    double beta = 0.5;
    double bias = 1;
    size_t size = 3;
    auto lrn = make_shared<op::LRN>(A, axes, alpha, beta, bias, size);
    auto f = make_shared<Function>(lrn, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    vector<float> args{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, args);

    auto result = backend->create_tensor(element::f32, shape);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});

    vector<float> expected{0.0f,
                           0.7071068f,
                           0.5345225f,
                           0.8017837f,
                           0.6172134f,
                           0.7715167f,
                           0.6469966f,
                           0.7548294f,
                           0.6620847f,
                           0.7448453f,
                           0.671156f,
                           0.7382717f};
    EXPECT_TRUE(test::all_close_f(expected, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, lrn_across_hw)
{
    Shape shape{2, 3, 2, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axes = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{2, 3});
    double alpha = 3;
    double beta = 0.5;
    double bias = 1;
    size_t size = 3;
    auto lrn = make_shared<op::LRN>(A, axes, alpha, beta, bias, size);
    auto f = make_shared<Function>(lrn, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    vector<float> args{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, args);

    auto result = backend->create_tensor(element::f32, shape);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});

    vector<float> expected{0.0f,
                           0.7071068f,
                           0.5345225f,
                           0.8017837f,
                           0.6172134f,
                           0.7715167f,
                           0.6469966f,
                           0.7548294f,
                           0.6620847f,
                           0.7448453f,
                           0.671156f,
                           0.7382717f};
    EXPECT_TRUE(test::all_close_f(expected, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, lrn_across_all_dims)
{
    Shape shape{2, 3, 2, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axes = make_shared<op::Constant>(element::i64, Shape{4}, vector<int64_t>{0, 1, 2, 3});
    double alpha = 3;
    double beta = 0.5;
    double bias = 1;
    size_t size = 3;
    auto lrn = make_shared<op::LRN>(A, axes, alpha, beta, bias, size);
    auto f = make_shared<Function>(lrn, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    vector<float> args{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, args);

    auto result = backend->create_tensor(element::f32, shape);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});

    vector<float> expected{0.0f,
                           0.0638877f,
                           0.0888231f,
                           0.1332347f,
                           0.1949481f,
                           0.2436851f,
                           0.3833259f,
                           0.4472136f,
                           0.3552925f,
                           0.399704f,
                           0.4873702f,
                           0.5361072f};
    EXPECT_TRUE(
        test::all_close_f(expected, read_vector<float>(result), DEFAULT_FLOAT_TOLERANCE_BITS + 1));
}

NGRAPH_TEST(${BACKEND_NAME}, lrn_across_nw)
{
    Shape shape{2, 3, 2, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axes = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{0, 3});
    double alpha = 3;
    double beta = 0.5;
    double bias = 1;
    size_t size = 3;
    auto lrn = make_shared<op::LRN>(A, axes, alpha, beta, bias, size);
    auto f = make_shared<Function>(lrn, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    vector<float> args{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, args);

    auto result = backend->create_tensor(element::f32, shape);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});

    vector<float> expected{0.0f,
                           0.140028f,
                           0.2407717f,
                           0.3144855f,
                           0.3698001f,
                           0.4123931f,
                           0.9863939f,
                           0.9801961f,
                           0.9630868f,
                           0.9434564f,
                           0.9245003f,
                           0.9072647f};
    EXPECT_TRUE(test::all_close_f(expected, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, lrn_across_empty)
{
    Shape shape{2, 3, 2, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axes = make_shared<op::Constant>(element::i64, Shape{0}, vector<int64_t>{});
    double alpha = 3;
    double beta = 0.5;
    double bias = 1;
    size_t size = 3;
    auto lrn = make_shared<op::LRN>(A, axes, alpha, beta, bias, size);
    auto f = make_shared<Function>(lrn, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    vector<float> args{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, args);

    auto result = backend->create_tensor(element::f32, shape);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});

    vector<float> expected{
        0.0f,
        0.7071068f,
        0.8944272f,
        0.9486833f,
        0.9701425f,
        0.9805807f,
        0.9863939f,
        0.9899495f,
        0.9922779f,
        0.9938837f,
        0.9950372f,
        0.9958932f,
    };
    EXPECT_TRUE(test::all_close_f(expected, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, lrn_6D_across_2_axes)
{
    Shape shape{2, 3, 2, 2, 1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axes = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{2, 3});
    double alpha = 3;
    double beta = 0.5;
    double bias = 1;
    size_t size = 3;
    auto lrn = make_shared<op::LRN>(A, axes, alpha, beta, bias, size);
    auto f = make_shared<Function>(lrn, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    vector<float> args(24);
    std::iota(std::begin(args), std::end(args), 0);
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, args);

    auto result = backend->create_tensor(element::f32, shape);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});

    vector<float> expected{0.0f,       0.2581989f, 0.5163978f, 0.7745967f, 0.3549426f, 0.4436783f,
                           0.5324139f, 0.6211495f, 0.4175966f, 0.4697962f, 0.5219957f, 0.5741953f,
                           0.4426267f, 0.4795122f, 0.5163978f, 0.5532833f, 0.4560274f, 0.4845291f,
                           0.5130308f, 0.5415326f, 0.4643635f, 0.4875816f, 0.5107998f, 0.534018f};
    EXPECT_TRUE(test::all_close_f(expected, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, lrn_2d_across_empty)
{
    Shape shape{12};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axes = make_shared<op::Constant>(element::i64, Shape{0}, vector<int64_t>{});
    double alpha = 3;
    double beta = 0.5;
    double bias = 1;
    size_t size = 3;
    auto lrn = make_shared<op::LRN>(A, axes, alpha, beta, bias, size);
    auto f = make_shared<Function>(lrn, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    vector<float> args{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, args);

    auto result = backend->create_tensor(element::f32, shape);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});

    vector<float> expected{
        0.0f,
        0.7071068f,
        0.8944272f,
        0.9486833f,
        0.9701425f,
        0.9805807f,
        0.9863939f,
        0.9899495f,
        0.9922779f,
        0.9938837f,
        0.9950372f,
        0.9958932f,
    };
    EXPECT_TRUE(test::all_close_f(expected, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, lrn_2d_across_outermost_axis)
{
    Shape shape{6, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axes = make_shared<op::Constant>(element::i64, Shape{1}, vector<int64_t>{0});
    double alpha = 0.0002;
    double beta = 0.5;
    double bias = 2.0;
    size_t size = 3;
    auto lrn = make_shared<op::LRN>(A, axes, alpha, beta, bias, size);
    auto f = make_shared<Function>(lrn, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    vector<float> args{0.64915806f,
                       0.21213771f,
                       -1.48256505f,
                       -1.41040838f,
                       0.58189541f,
                       0.11432108f,
                       -0.22993855f,
                       -0.13325502f,
                       -0.03083259f,
                       -0.48450908f,
                       0.50342429f,
                       -0.99551708f};
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, args);

    auto result = backend->create_tensor(element::f32, shape);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});

    vector<float> expected{0.45900404f,
                           0.14999892f,
                           -1.04828012f,
                           -0.99727529f,
                           0.41144446f,
                           0.08083449f,
                           -0.16259004f,
                           -0.09422511f,
                           -0.02180192f,
                           -0.34259823f,
                           0.35597473f,
                           -0.70393407f};
    EXPECT_TRUE(test::all_close_f(expected, read_vector<float>(result), 23));
}
