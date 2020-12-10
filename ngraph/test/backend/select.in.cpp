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

NGRAPH_TEST(${BACKEND_NAME}, select)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Select>(A, B, C), ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::boolean, shape);
    copy_data(a, vector<char>{0, 1, 1, 0, 0, 1, 0, 1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    auto c = backend->create_tensor(element::f32, shape);
    copy_data(c, vector<float>{11, 12, 13, 14, 15, 16, 17, 18});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b, c});
    EXPECT_TRUE(test::all_close_f((vector<float>{11, 2, 3, 14, 15, 6, 17, 8}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, select_v1)
{
    auto A = make_shared<op::Parameter>(element::boolean, Shape{4});
    auto B = make_shared<op::Parameter>(element::f32, Shape{4});
    auto C = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto f = make_shared<Function>(make_shared<op::v1::Select>(A, B, C), ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::boolean, Shape{4});
    copy_data(a, vector<char>{0, 1, 1, 0});
    auto b = backend->create_tensor(element::f32, Shape{4});
    copy_data(b, vector<float>{1, 2, 3, 4});
    auto c = backend->create_tensor(element::f32, Shape{2, 4});
    copy_data(c, vector<float>{11, 12, 13, 14, 15, 16, 17, 18});
    auto result = backend->create_tensor(element::f32, Shape{2, 4});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b, c});
    EXPECT_TRUE(
        test::all_close_f((vector<float>{11, 2, 3, 14, 15, 2, 3, 18}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, select_double)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto B = make_shared<op::Parameter>(element::f64, shape);
    auto C = make_shared<op::Parameter>(element::f64, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Select>(A, B, C), ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::boolean, shape);
    copy_data(a, vector<char>{0, 1, 1, 0, 0, 1, 0, 1});
    auto b = backend->create_tensor(element::f64, shape);
    copy_data(b, vector<double>{1, 2, 3, 4, 5, 6, 7, 8});
    auto c = backend->create_tensor(element::f64, shape);
    copy_data(c, vector<double>{11, 12, 13, 14, 15, 16, 17, 18});
    auto result = backend->create_tensor(element::f64, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b, c});
    EXPECT_TRUE(test::all_close_f((vector<double>{11, 2, 3, 14, 15, 6, 17, 8}),
                                  read_vector<double>(result)));
}
