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

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset7.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/shape.hpp"
#include "runtime/backend.hpp"
#include "util/all_close_f.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, roll_2d_input)
{
    Shape shape{4, 3};
    auto x = make_shared<opset7::Parameter>(element::f32, shape);
    auto shift = make_shared<opset7::Constant>(element::i64, Shape{1}, vector<int64_t>{1});
    auto axes = make_shared<opset7::Constant>(element::i64, Shape{1}, vector<int64_t>{0});
    auto f = make_shared<Function>(make_shared<opset7::Roll>(x, shift, axes), ParameterVector{x});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto x_tensor = backend->create_tensor(element::f32, shape);
    copy_data(x_tensor, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {x_tensor});
    EXPECT_TRUE(test::all_close_f((vector<float>{10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, roll_2d_input_negative_shift)
{
    Shape shape{4, 3};
    auto x = make_shared<opset7::Parameter>(element::f32, shape);
    auto shift = make_shared<opset7::Constant>(element::i64, Shape{2}, vector<int64_t>{-1, 2});
    auto axes = make_shared<opset7::Constant>(element::i64, Shape{2}, vector<int64_t>{0, 1});
    auto f = make_shared<Function>(make_shared<opset7::Roll>(x, shift, axes), ParameterVector{x});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto x_tensor = backend->create_tensor(element::f32, shape);
    copy_data(x_tensor, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {x_tensor});
    EXPECT_TRUE(test::all_close_f((vector<float>{5, 6, 4, 8, 9, 7, 11, 12, 10, 2, 3, 1}),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, roll_repeated_axes)
{
    Shape shape{4, 3};
    auto x = make_shared<opset7::Parameter>(element::f32, shape);
    auto shift = make_shared<opset7::Constant>(element::i64, Shape{3}, vector<int64_t>{1, 2, 1});
    auto axes = make_shared<opset7::Constant>(element::i64, Shape{3}, vector<int64_t>{0, 1, 0});
    auto f = make_shared<Function>(make_shared<opset7::Roll>(x, shift, axes), ParameterVector{x});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto x_tensor = backend->create_tensor(element::f32, shape);
    copy_data(x_tensor, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {x_tensor});
    EXPECT_TRUE(test::all_close_f((vector<float>{8, 9, 7, 11, 12, 10, 2, 3, 1, 5, 6, 4}),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, roll_3d_input)
{
    Shape shape{4, 2, 3};
    auto x = make_shared<opset7::Parameter>(element::f32, shape);
    auto shift = make_shared<opset7::Constant>(element::i64, Shape{3}, vector<int64_t>{2, 1, 3});
    auto axes = make_shared<opset7::Constant>(element::i64, Shape{3}, vector<int64_t>{0, 1, 2});
    auto f = make_shared<Function>(make_shared<opset7::Roll>(x, shift, axes), ParameterVector{x});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x_tensor = backend->create_tensor(element::f32, shape);
    copy_data(x_tensor, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                      13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    auto result = backend->create_tensor(element::f32, shape);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {x_tensor});
    EXPECT_TRUE(test::all_close_f((vector<float>{16, 17, 18, 13, 14, 15, 22, 23, 24, 19, 20, 21,
                                                 4,  5,  6,  1,  2,  3,  10, 11, 12, 7,  8,  9}),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, roll_3d_input_negative_shift)
{
    Shape shape{4, 2, 3};
    auto x = make_shared<opset7::Parameter>(element::f32, shape);
    auto shift = make_shared<opset7::Constant>(element::i64, Shape{3}, vector<int64_t>{-5, 1, 3});
    auto axes = make_shared<opset7::Constant>(element::i64, Shape{3}, vector<int64_t>{0, 1, 1});
    auto f = make_shared<Function>(make_shared<opset7::Roll>(x, shift, axes), ParameterVector{x});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x_tensor = backend->create_tensor(element::f32, shape);
    copy_data(x_tensor, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                      13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    auto result = backend->create_tensor(element::f32, shape);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {x_tensor});
    EXPECT_TRUE(test::all_close_f((vector<float>{7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
                                                 19, 20, 21, 22, 23, 24, 1,  2,  3,  4,  5,  6}),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, roll_negative_axes)
{
    Shape shape{4, 2, 3};
    auto x = make_shared<opset7::Parameter>(element::f32, shape);
    auto shift = make_shared<opset7::Constant>(element::i64, Shape{3}, vector<int64_t>{2, -1, -7});
    auto axes = make_shared<opset7::Constant>(element::i64, Shape{3}, vector<int64_t>{-1, -1, -2});
    auto f = make_shared<Function>(make_shared<opset7::Roll>(x, shift, axes), ParameterVector{x});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x_tensor = backend->create_tensor(element::f32, shape);
    copy_data(x_tensor, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                      13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    auto result = backend->create_tensor(element::f32, shape);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {x_tensor});
    EXPECT_TRUE(test::all_close_f((vector<float>{6,  4,  5,  3,  1,  2,  12, 10, 11, 9,  7,  8,
                                                 18, 16, 17, 15, 13, 14, 24, 22, 23, 21, 19, 20}),
                                  read_vector<float>(result)));
}
