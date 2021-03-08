//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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
#include "util/random.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

static std::mt19937_64 random_generator;

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, cum_sum_default)
{
    Shape shape{1, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axis = make_shared<op::Parameter>(element::i32, Shape{1});
    auto f = make_shared<Function>(make_shared<op::CumSum>(A, axis), ParameterVector{A, axis});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto axis_tensor = backend->create_tensor(axis->get_element_type(), axis->get_shape());
    copy_data(axis_tensor, vector<int32_t>{1});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, axis_tensor});
    EXPECT_TRUE(test::all_close_f((vector<float>{1, 3, 6, 10}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, cum_sum_2dim)
{
    Shape shape{2, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axis = make_shared<op::Parameter>(element::i64, Shape{1});
    auto f = make_shared<Function>(make_shared<op::CumSum>(A, axis), ParameterVector{A, axis});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7});
    auto axis_tensor = backend->create_tensor(axis->get_element_type(), axis->get_shape());
    copy_data(axis_tensor, vector<int64_t>{0});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, axis_tensor});
    EXPECT_TRUE(
        test::all_close_f((vector<float>{0, 1, 2, 3, 4, 6, 8, 10}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, cum_sum_2dim_default_axis)
{
    Shape shape{2, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::CumSum>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(
        test::all_close_f((vector<float>{0, 1, 2, 3, 4, 6, 8, 10}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, cum_sum_3d)
{
    auto test_cumsum_3d = [](const int32_t axis_val) -> void {
        Shape shape{3, 2, 4};
        auto A = make_shared<op::Parameter>(element::f32, shape);
        auto axis = make_shared<op::Parameter>(element::i32, Shape{1});
        auto f = make_shared<Function>(make_shared<op::CumSum>(A, axis), ParameterVector{A, axis});

        auto backend = runtime::Backend::create("${BACKEND_NAME}");

        // Create some tensors for input/output
        auto a = backend->create_tensor(element::f32, shape);
        copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                   12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
        auto axis_tensor = backend->create_tensor(axis->get_element_type(), axis->get_shape());
        copy_data(axis_tensor, vector<int32_t>{axis_val});
        auto result = backend->create_tensor(element::f32, shape);

        auto handle = backend->compile(f);
        handle->call_with_validate({result}, {a, axis_tensor});

        if (axis_val == 0)
        {
            EXPECT_TRUE(
                test::all_close_f((vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  10, 12, 14,
                                                 16, 18, 20, 22, 24, 27, 30, 33, 36, 39, 42, 45}),
                                  read_vector<float>(result)));
        }
        else if (axis_val == 1)
        {
            EXPECT_TRUE(
                test::all_close_f((vector<float>{0,  1,  2,  3,  4,  6,  8,  10, 8,  9,  10, 11,
                                                 20, 22, 24, 26, 16, 17, 18, 19, 36, 38, 40, 42}),
                                  read_vector<float>(result)));
        }
        else if (axis_val == 2)
        {
            EXPECT_TRUE(
                test::all_close_f((vector<float>{0,  1,  3,  6,  4,  9,  15, 22, 8,  17, 27, 38,
                                                 12, 25, 39, 54, 16, 33, 51, 70, 20, 41, 63, 86}),
                                  read_vector<float>(result)));
        }
    };
    test_cumsum_3d(0);
    test_cumsum_3d(1);
    test_cumsum_3d(2);
}

NGRAPH_TEST(${BACKEND_NAME}, cum_sum_2dim_allmodes)
{
    auto test_cum_sum_allmodes = [](const int64_t axis_val, int exclusive, int reverse) {
        Shape shape{2, 4};
        auto A = make_shared<op::Parameter>(element::f32, shape);
        auto axis = make_shared<op::Parameter>(element::i64, Shape{1});
        auto f = make_shared<Function>(make_shared<op::CumSum>(A, axis, exclusive, reverse),
                                       ParameterVector{A, axis});

        auto backend = runtime::Backend::create("${BACKEND_NAME}");

        // Create some tensors for input/output
        auto a = backend->create_tensor(element::f32, shape);
        copy_data(a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7});
        auto axis_tensor = backend->create_tensor(axis->get_element_type(), axis->get_shape());
        copy_data(axis_tensor, vector<int64_t>{axis_val});
        auto result = backend->create_tensor(element::f32, shape);

        auto handle = backend->compile(f);
        handle->call_with_validate({result}, {a, axis_tensor});
        if (axis_val == 1 && exclusive == 1 && reverse == 0)
        {
            EXPECT_TRUE(test::all_close_f((vector<float>{0, 0, 1, 3, 0, 4, 9, 15}),
                                          read_vector<float>(result)));
        }
        else if (axis_val == 1 && exclusive == 0 && reverse == 1)
        {
            EXPECT_TRUE(test::all_close_f((vector<float>{6, 6, 5, 3, 22, 18, 13, 7}),
                                          read_vector<float>(result)));
        }
        else if (axis_val == 1 && exclusive == 1 && reverse == 1)
        {
            EXPECT_TRUE(test::all_close_f((vector<float>{6, 5, 3, 0, 18, 13, 7, 0}),
                                          read_vector<float>(result)));
        }
        else if (axis_val == 0 && exclusive == 0 && reverse == 0)
        {
            EXPECT_TRUE(test::all_close_f((vector<float>{0, 1, 2, 3, 4, 6, 8, 10}),
                                          read_vector<float>(result)));
        }
        else if (axis_val == 0 && exclusive == 1 && reverse == 1)
        {
            EXPECT_TRUE(test::all_close_f((vector<float>{4, 5, 6, 7, 0, 0, 0, 0}),
                                          read_vector<float>(result)));
        }
        else if (axis_val == 0 && exclusive == 0 && reverse == 1)
        {
            EXPECT_TRUE(test::all_close_f((vector<float>{4, 6, 8, 10, 4, 5, 6, 7}),
                                          read_vector<float>(result)));
        }
    };

    test_cum_sum_allmodes(1, 1, 0);
    test_cum_sum_allmodes(-1, 0, 1);
    test_cum_sum_allmodes(-1, 1, 1);
    test_cum_sum_allmodes(0, 0, 0);
    test_cum_sum_allmodes(0, 1, 1);
    test_cum_sum_allmodes(0, 0, 1);
}
