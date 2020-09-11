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
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, tensor_constant)
{
    Shape shape{2, 2, 2};
    auto A = op::Constant::create(element::f32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto f = make_shared<Function>(A, ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {});
    EXPECT_TRUE(test::all_close_f((vector<float>{1, 2, 3, 4, 5, 6, 7, 8}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, tensor_2constant)
{
    Shape shape{2, 2, 2};
    auto A = op::Constant::create(element::f32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto f = make_shared<Function>(NodeVector{A, A}, ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result0 = backend->create_tensor(element::f32, shape);
    auto result1 = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result0, result1}, {});
    EXPECT_TRUE(test::all_close_f((vector<float>{1, 2, 3, 4, 5, 6, 7, 8}),
                                  read_vector<float>(result0),
                                  MIN_FLOAT_TOLERANCE_BITS));
    EXPECT_TRUE(test::all_close_f((vector<float>{1, 2, 3, 4, 5, 6, 7, 8}),
                                  read_vector<float>(result1),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, tensor_constant_with_op)
{
    Shape shape{2, 2, 2};
    auto A = op::Constant::create(element::f32, shape, {-1, 2, 3, -4, 5, -6, -7, 8});
    auto f = make_shared<Function>(make_shared<op::Abs>(A), ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {});
    EXPECT_TRUE(test::all_close_f((vector<float>{1, 2, 3, 4, 5, 6, 7, 8}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, constant_multi_use)
{
    auto A = make_shared<op::Constant>(element::i32, Shape{}, std::vector<std::string>{"388"});
    auto f = make_shared<Function>(A, ParameterVector{});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    std::shared_ptr<runtime::Tensor> r1 = backend->create_tensor(element::i32, Shape{});
    auto handle = backend->compile(f);
    handle->call_with_validate({r1}, std::vector<std::shared_ptr<runtime::Tensor>>{});
    EXPECT_EQ(read_vector<int>(r1), std::vector<int>{388});

    std::shared_ptr<runtime::Tensor> r2 = backend->create_tensor(element::i32, Shape{});
    handle->call_with_validate({r2}, std::vector<std::shared_ptr<runtime::Tensor>>{});
    EXPECT_EQ(read_vector<int>(r2), std::vector<int>{388});
}

NGRAPH_TEST(${BACKEND_NAME}, scalar_constant_float32)
{
    auto r = op::Constant::create(element::f32, Shape{}, {4.75});
    auto f = make_shared<Function>(r, ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::f32, Shape{});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {});
    EXPECT_TRUE(test::all_close_f(
        vector<float>{4.75f}, read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, scalar_constant_int64)
{
    auto r = op::Constant::create(element::i64, Shape{}, {0x4000000000000001});
    auto f = make_shared<Function>(r, ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::i64, Shape{});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {});
    EXPECT_EQ(vector<int64_t>{0x4000000000000001}, read_vector<int64_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, tensor_constant_float32)
{
    Shape shape{2, 2};
    auto r = op::Constant::create(element::f32, shape, {4.75, 4.5, -5.25, 0.0});
    auto f = make_shared<Function>(r, ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {});
    EXPECT_TRUE(test::all_close_f((vector<float>{4.75f, 4.5f, -5.25f, 0.0f}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, tensor_constant_int64)
{
    Shape shape{2};
    auto r = op::Constant::create(element::i64, shape, {0x4000000000000001, 0x4000000000000002});
    auto f = make_shared<Function>(r, ParameterVector{});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    // Create some tensors for input/output
    auto result = backend->create_tensor(element::i64, shape);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {});
    EXPECT_EQ((vector<int64_t>{0x4000000000000001, 0x4000000000000002}),
              read_vector<int64_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, constant_equality_bool)
{
    Shape shape{4};
    // auto A = make_shared<op::Parameter>(element::boolean, shape);
    // auto B = make_shared<op::Parameter>(element::boolean, shape);
    // auto f = make_shared<Function>(make_shared<op::Equal>(A, B), ParameterVector{A, B});

    auto A = op::Constant::create(element::boolean, shape, {true, false, true, false});
    auto B = op::Constant::create(element::boolean, shape, {true, true, true, true});
    auto f = make_shared<Function>(make_shared<op::Equal>(A, B), ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::boolean, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {});
    EXPECT_EQ((vector<char>{true, false, true, false}), read_vector<char>(result));
}
