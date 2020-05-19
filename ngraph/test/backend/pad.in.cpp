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

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "runtime/backend.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/known_element_types.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, pad_exterior_1d)
{
    Shape shape_a{6};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{15};
    CoordinateDiff padding_below{4};
    CoordinateDiff padding_above{5};
    auto f = make_shared<Function>(make_shared<op::Pad>(A, B, padding_below, padding_above),
                                   ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f(
        (test::NDArray<float, 1>(
             {2112, 2112, 2112, 2112, 1, 2, 3, 4, 5, 6, 2112, 2112, 2112, 2112, 2112})
             .get_vector()),
        read_vector<float>(result),
        MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_negative_exterior_1d)
{
    Shape shape_a{6};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{8};
    CoordinateDiff padding_below{4};
    CoordinateDiff padding_above{-2};
    auto f = make_shared<Function>(make_shared<op::Pad>(A, B, padding_below, padding_above),
                                   ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f(
        (test::NDArray<float, 1>({2112, 2112, 2112, 2112, 1, 2, 3, 4}).get_vector()),
        read_vector<float>(result),
        MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_negative_exterior_1d_check_limits)
{
    Shape shape_a{6};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{3};
    CoordinateDiff padding_below{4};
    CoordinateDiff padding_above{-7};
    auto f = make_shared<Function>(make_shared<op::Pad>(A, B, padding_below, padding_above),
                                   ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((test::NDArray<float, 1>({2112, 2112, 2112}).get_vector()),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_edge_1d)
{
    Shape shape_a{6};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{11};
    CoordinateDiff padding_below{2};
    CoordinateDiff padding_above{3};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, op::PadMode::EDGE),
        ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(
        test::all_close_f((test::NDArray<float, 1>({1, 1, 1, 2, 3, 4, 5, 6, 6, 6, 6}).get_vector()),
                          read_vector<float>(result),
                          MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_edge_1d_top_neg)
{
    Shape shape_a{6};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{5};
    CoordinateDiff padding_below{2};
    CoordinateDiff padding_above{-3};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, op::PadMode::EDGE),
        ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((test::NDArray<float, 1>({1, 1, 1, 2, 3}).get_vector()),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_edge_1d_top_neg_bigger_than_tensor)
{
    Shape shape_a{6};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{1};
    CoordinateDiff padding_below{2};
    CoordinateDiff padding_above{-7};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, op::PadMode::EDGE),
        ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((test::NDArray<float, 1>({1}).get_vector()),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_edge_1d_bottom_neg)
{
    Shape shape_a{6};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{7};
    CoordinateDiff padding_below{-2};
    CoordinateDiff padding_above{3};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, op::PadMode::EDGE),
        ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((test::NDArray<float, 1>({3, 4, 5, 6, 6, 6, 6}).get_vector()),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_edge_1d_bottom_neg_bigger_than_tensor)
{
    Shape shape_a{6};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{2};
    CoordinateDiff padding_below{-7};
    CoordinateDiff padding_above{3};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, op::PadMode::EDGE),
        ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((test::NDArray<float, 1>({6, 6}).get_vector()),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_edge_2d)
{
    Shape shape_a{3, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{6, 9};
    CoordinateDiff padding_below{2, 3};
    CoordinateDiff padding_above{1, 2};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, op::PadMode::EDGE),
        ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 2>({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((test::NDArray<float, 2>({{1, 1, 1, 1, 2, 3, 4, 4, 4},
                                                            {1, 1, 1, 1, 2, 3, 4, 4, 4},
                                                            {1, 1, 1, 1, 2, 3, 4, 4, 4},
                                                            {5, 5, 5, 5, 6, 7, 8, 8, 8},
                                                            {9, 9, 9, 9, 10, 11, 12, 12, 12},
                                                            {9, 9, 9, 9, 10, 11, 12, 12, 12}})
                                       .get_vector()),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_edge_2d_with_neg)
{
    Shape shape_a{3, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{6, 5};
    CoordinateDiff padding_below{2, -1};
    CoordinateDiff padding_above{1, 2};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, op::PadMode::EDGE),
        ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 2>({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((test::NDArray<float, 2>({{2, 3, 4, 4, 4},
                                                            {2, 3, 4, 4, 4},
                                                            {2, 3, 4, 4, 4},
                                                            {6, 7, 8, 8, 8},
                                                            {10, 11, 12, 12, 12},
                                                            {10, 11, 12, 12, 12}})
                                       .get_vector()),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_reflect_1d)
{
    Shape shape_a{6};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{11};
    CoordinateDiff padding_below{2};
    CoordinateDiff padding_above{3};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, op::PadMode::REFLECT),
        ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(
        test::all_close_f((test::NDArray<float, 1>({3, 2, 1, 2, 3, 4, 5, 6, 5, 4, 3}).get_vector()),
                          read_vector<float>(result),
                          MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_reflect_1d_top_neg)
{
    Shape shape_a{6};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{5};
    CoordinateDiff padding_below{2};
    CoordinateDiff padding_above{-3};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, op::PadMode::REFLECT),
        ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((test::NDArray<float, 1>({3, 2, 1, 2, 3}).get_vector()),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_reflect_1d_top_neg_bigger_than_tensor)
{
    Shape shape_a{6};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{1};
    CoordinateDiff padding_below{2};
    CoordinateDiff padding_above{-7};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, op::PadMode::REFLECT),
        ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((test::NDArray<float, 1>({3}).get_vector()),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_reflect_1d_bottom_neg)
{
    Shape shape_a{6};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{7};
    CoordinateDiff padding_below{-2};
    CoordinateDiff padding_above{3};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, op::PadMode::REFLECT),
        ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((test::NDArray<float, 1>({3, 4, 5, 6, 5, 4, 3}).get_vector()),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_reflect_1d_bottom_neg_bigger_than_tensor)
{
    Shape shape_a{6};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{2};
    CoordinateDiff padding_below{-7};
    CoordinateDiff padding_above{3};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, op::PadMode::REFLECT),
        ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((test::NDArray<float, 1>({4, 3}).get_vector()),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_reflect_1d_multi_reflect)
{
    Shape shape_a{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{22};
    CoordinateDiff padding_below{10};
    CoordinateDiff padding_above{9};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, op::PadMode::REFLECT),
        ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, test::NDArray<float, 1>({1, 2, 3}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f(
        (test::NDArray<float, 1>({3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2})
             .get_vector()),
        read_vector<float>(result),
        MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_reflect_2d)
{
    Shape shape_a{3, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{6, 9};
    CoordinateDiff padding_below{2, 3};
    CoordinateDiff padding_above{1, 2};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, op::PadMode::REFLECT),
        ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 2>({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((test::NDArray<float, 2>({{12, 11, 10, 9, 10, 11, 12, 11, 10},
                                                            {8, 7, 6, 5, 6, 7, 8, 7, 6},
                                                            {4, 3, 2, 1, 2, 3, 4, 3, 2},
                                                            {8, 7, 6, 5, 6, 7, 8, 7, 6},
                                                            {12, 11, 10, 9, 10, 11, 12, 11, 10},
                                                            {8, 7, 6, 5, 6, 7, 8, 7, 6}})
                                       .get_vector()),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_reflect_2d_with_neg)
{
    Shape shape_a{3, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{6, 5};
    CoordinateDiff padding_below{2, -1};
    CoordinateDiff padding_above{1, 2};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, op::PadMode::REFLECT),
        ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 2>({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});

    EXPECT_TRUE(test::all_close_f((test::NDArray<float, 2>({{10, 11, 12, 11, 10},
                                                            {6, 7, 8, 7, 6},
                                                            {2, 3, 4, 3, 2},
                                                            {6, 7, 8, 7, 6},
                                                            {10, 11, 12, 11, 10},
                                                            {6, 7, 8, 7, 6}})
                                       .get_vector()),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_negative_exterior_2d)
{
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{5, 2};
    CoordinateDiff padding_below{1, -1};
    CoordinateDiff padding_above{2, 0};
    auto f = make_shared<Function>(make_shared<op::Pad>(A, B, padding_below, padding_above),
                                   ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, test::NDArray<float, 2>({{1, 2, 3}, {4, 5, 6}}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{9});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f(
        (test::NDArray<float, 2>({{9, 9}, {2, 3}, {5, 6}, {9, 9}, {9, 9}}).get_vector()),
        read_vector<float>(result),
        MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_negative_exterior_2d_all_negative)
{
    Shape shape_a{3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{1, 1};
    CoordinateDiff padding_below{-1, -1};
    CoordinateDiff padding_above{-1, -1};
    auto f = make_shared<Function>(make_shared<op::Pad>(A, B, padding_below, padding_above),
                                   ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, test::NDArray<float, 2>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{9});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((test::NDArray<float, 2>({{5}}).get_vector()),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_exterior_2d_0x0)
{
    Shape shape_a{0, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{5, 5};
    CoordinateDiff padding_below{2, 3};
    CoordinateDiff padding_above{3, 2};
    auto f = make_shared<Function>(make_shared<op::Pad>(A, B, padding_below, padding_above),
                                   ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    // copy_data(a, test::NDArray<float, 2>({{}}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((test::NDArray<float, 2>({{2112, 2112, 2112, 2112, 2112},
                                                            {2112, 2112, 2112, 2112, 2112},
                                                            {2112, 2112, 2112, 2112, 2112},
                                                            {2112, 2112, 2112, 2112, 2112},
                                                            {2112, 2112, 2112, 2112, 2112}})
                                       .get_vector()),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_exterior_2d_0x3)
{
    Shape shape_a{0, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{5, 5};
    CoordinateDiff padding_below{2, 1};
    CoordinateDiff padding_above{3, 1};
    auto f = make_shared<Function>(make_shared<op::Pad>(A, B, padding_below, padding_above),
                                   ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    // copy_data(a, test::NDArray<float, 2>({}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((test::NDArray<float, 2>({{2112, 2112, 2112, 2112, 2112},
                                                            {2112, 2112, 2112, 2112, 2112},
                                                            {2112, 2112, 2112, 2112, 2112},
                                                            {2112, 2112, 2112, 2112, 2112},
                                                            {2112, 2112, 2112, 2112, 2112}})
                                       .get_vector()),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_exterior_2d_3x0)
{
    Shape shape_a{3, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{5, 5};
    CoordinateDiff padding_below{1, 3};
    CoordinateDiff padding_above{1, 2};
    auto f = make_shared<Function>(make_shared<op::Pad>(A, B, padding_below, padding_above),
                                   ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    // copy_data(a, test::NDArray<float, 2>({}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((test::NDArray<float, 2>({{2112, 2112, 2112, 2112, 2112},
                                                            {2112, 2112, 2112, 2112, 2112},
                                                            {2112, 2112, 2112, 2112, 2112},
                                                            {2112, 2112, 2112, 2112, 2112},
                                                            {2112, 2112, 2112, 2112, 2112}})
                                       .get_vector()),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_exterior_4d_1x2x2x2)
{
    Shape shape_a{1, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{1, 2, 4, 4};
    CoordinateDiff padding_below{0, 0, 1, 1};
    CoordinateDiff padding_above{0, 0, 1, 1};
    auto f = make_shared<Function>(make_shared<op::Pad>(A, B, padding_below, padding_above),
                                   ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    // clang-format off
    copy_data(a, test::NDArray<float, 4>(
        {
            {
                {
                    {0.0f, 0.0f},
                    {0.0f, 0.0f}
                },
                {
                    {0.0f, 0.0f},
                    {0.0f, 0.0f}
                }
            }
        }).get_vector());
    // clang-format on

    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{42});

    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    // clang-format off
    EXPECT_TRUE(test::all_close_f((test::NDArray<float, 4>(
        {
            {
                {
                    {42.0f, 42.0f, 42.0f, 42.0f},
                    {42.0f, 0.0f, 0.0f, 42.0f},
                    {42.0f, 0.0f, 0.0f, 42.0f},
                    {42.0f, 42.0f, 42.0f, 42.0f}
                },
                {
                    {42.0f, 42.0f, 42.0f, 42.0f},
                    {42.0f, 0.0f, 0.0f, 42.0f},
                    {42.0f, 0.0f, 0.0f, 42.0f},
                    {42.0f, 42.0f, 42.0f, 42.0f}
                }
            }
        }).get_vector()),
        read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
    // clang-format on
}

NGRAPH_TEST(${BACKEND_NAME}, pad_negative_exterior_4d)
{
    Shape shape_a{1, 3, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{1, 1, 4, 4};
    CoordinateDiff padding_below{0, -1, 1, 1};
    CoordinateDiff padding_above{0, -1, 1, 1};
    auto f = make_shared<Function>(make_shared<op::Pad>(A, B, padding_below, padding_above),
                                   ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    // clang-format off
    copy_data(a, test::NDArray<float, 4>(
        {
            {
                {
                    {0.0f, 0.0f},
                    {0.0f, 0.0f}
                },
                {
                    {1.0f, 1.0f},
                    {1.0f, 1.0f}
                },
                {
                    {2.0f, 2.0f},
                    {2.0f, 2.0f}
                }
            }
        }).get_vector());
    // clang-format on

    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{42});

    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    // clang-format off
    EXPECT_TRUE(test::all_close_f((test::NDArray<float, 4>(
        {
            {
                {
                    {42.0f, 42.0f, 42.0f, 42.0f},
                    {42.0f, 1.0f, 1.0f, 42.0f},
                    {42.0f, 1.0f, 1.0f, 42.0f},
                    {42.0f, 42.0f, 42.0f, 42.0f}
                }
            }
        }).get_vector()),
        read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
    // clang-format on
}

// This test covers the case with multiple image and with asymetric pad
// bug has been found on nvGPU side now covered by this test
NGRAPH_TEST(${BACKEND_NAME}, pad_2channel_2image_asym)
{
    Shape shape_a{2, 2, 4, 4};
    auto window_movement_strides = Strides{2, 2};
    CoordinateDiff padding_below{0, 0, 0, 0};
    CoordinateDiff padding_above{0, 0, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{2, 2, 6, 6};
    auto f = make_shared<Function>(make_shared<op::Pad>(A, B, padding_below, padding_above),
                                   ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>({{{{0, 1, 0, 2}, // img 0 chan 0
                                         {0, 3, 2, 0},
                                         {2, 0, 0, 0},
                                         {0, 2, 1, 0}},

                                        {{0, 0, 0, 2}, // img 0 chan 1
                                         {0, 2, 3, 0},
                                         {2, 0, 1, 0},
                                         {2, 0, 0, 0}}},

                                       {{{0, 2, 1, 1}, // img 1 chan 0
                                         {0, 0, 2, 0},
                                         {0, 0, 1, 2},
                                         {0, 0, 0, 0}},

                                        {{2, 1, 0, 0}, // img 1 chan 1
                                         {0, 2, 0, 0},
                                         {1, 1, 2, 0},
                                         {1, 0, 0, 0}}}})
                  .get_vector());

    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{42});

    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((test::NDArray<float, 4>({{{{0, 1, 0, 2, 42, 42}, // img 0 chan 0
                                                              {0, 3, 2, 0, 42, 42},
                                                              {2, 0, 0, 0, 42, 42},
                                                              {0, 2, 1, 0, 42, 42},
                                                              {42, 42, 42, 42, 42, 42},
                                                              {42, 42, 42, 42, 42, 42}},

                                                             {{0, 0, 0, 2, 42, 42}, // img 1 chan 0
                                                              {0, 2, 3, 0, 42, 42},
                                                              {2, 0, 1, 0, 42, 42},
                                                              {2, 0, 0, 0, 42, 42},
                                                              {42, 42, 42, 42, 42, 42},
                                                              {42, 42, 42, 42, 42, 42}}},

                                                            {{{0, 2, 1, 1, 42, 42}, // img 1 chan 0
                                                              {0, 0, 2, 0, 42, 42},
                                                              {0, 0, 1, 2, 42, 42},
                                                              {0, 0, 0, 0, 42, 42},
                                                              {42, 42, 42, 42, 42, 42},
                                                              {42, 42, 42, 42, 42, 42}},

                                                             {{2, 1, 0, 0, 42, 42}, // img 1 chan 1
                                                              {0, 2, 0, 0, 42, 42},
                                                              {1, 1, 2, 0, 42, 42},
                                                              {1, 0, 0, 0, 42, 42},
                                                              {42, 42, 42, 42, 42, 42},
                                                              {42, 42, 42, 42, 42, 42}}}})
                                       .get_vector()),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_symmetric)
{
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{4, 7};
    CoordinateDiff padding_below{1, 2};
    CoordinateDiff padding_above{1, 2};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, op::PadMode::SYMMETRIC),
        ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, test::NDArray<float, 2>({{1, 2, 3}, {4, 5, 6}}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((test::NDArray<float, 2>({{2, 1, 1, 2, 3, 3, 2},
                                                            {2, 1, 1, 2, 3, 3, 2},
                                                            {5, 4, 4, 5, 6, 6, 5},
                                                            {5, 4, 4, 5, 6, 6, 5}})
                                       .get_vector()),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}
