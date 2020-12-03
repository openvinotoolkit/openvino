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

#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "runtime/backend.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, equal)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto B = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Equal>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0, 1, 1});
    auto b = backend->create_tensor(element::Type_t::f32, shape);
    copy_data(b, vector<float>{1, 8, 4, 8, 0, 0, 1, 1.5});
    auto result = backend->create_tensor(element::Type_t::boolean, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ((vector<char>{1, 1, 0, 0, 0, 1, 1, 0}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, notequal)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto B = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::NotEqual>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0, 1, 1});
    auto b = backend->create_tensor(element::Type_t::f32, shape);
    copy_data(b, vector<float>{1, 8, 4, 8, 0, 0, 1, 1.5});
    auto result = backend->create_tensor(element::Type_t::boolean, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ((vector<char>{0, 0, 1, 1, 1, 0, 0, 1}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, greater)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto B = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Greater>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0.5, 2, 1});
    auto b = backend->create_tensor(element::Type_t::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8, 0, 0, 1, 1.5});
    auto result = backend->create_tensor(element::Type_t::boolean, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ((vector<char>{0, 1, 0, 1, 0, 1, 1, 0}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, greater_int64)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Type_t::i64, shape);
    auto B = make_shared<op::Parameter>(element::Type_t::i64, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Greater>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::i64, shape);
    copy_data(a, vector<int64_t>{0x4000000000000002, 0x4000000000000006, -8, 17, -5, 5, 2, 1});
    auto b = backend->create_tensor(element::Type_t::i64, shape);
    copy_data(b, vector<int64_t>{0x4000000000000001, 0x4000000000000002, 4, 8, 0, 0, 1, 2});
    auto result = backend->create_tensor(element::Type_t::boolean, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ((vector<char>{1, 1, 0, 1, 0, 1, 1, 0}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, greatereq)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto B = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::GreaterEqual>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0, 2, 1});
    auto b = backend->create_tensor(element::Type_t::f32, shape);
    copy_data(b, vector<float>{1, 2, -8, 8, 0, 0, 0.5, 1.5});
    auto result = backend->create_tensor(element::Type_t::boolean, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ((vector<char>{1, 1, 1, 1, 0, 1, 1, 0}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, less)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto B = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Less>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0.5, 2, 1});
    auto b = backend->create_tensor(element::Type_t::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8, 0, 0, 1, 1.5});
    auto result = backend->create_tensor(element::Type_t::boolean, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ((vector<char>{0, 0, 1, 0, 1, 0, 0, 1}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, lesseq)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto B = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::LessEqual>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0, 2, 1});
    auto b = backend->create_tensor(element::Type_t::f32, shape);
    copy_data(b, vector<float>{1, 2, -8, 8, 0, 0, 0.5, 1.5});
    auto result = backend->create_tensor(element::Type_t::boolean, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ((vector<char>{1, 0, 1, 0, 1, 1, 0, 1}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, lesseq_int32)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Type_t::i32, shape);
    auto B = make_shared<op::Parameter>(element::Type_t::i32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::LessEqual>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::i32, shape);
    copy_data(a, vector<int32_t>{0x40000170, 0x40000005, 0x40000005, -5});
    auto b = backend->create_tensor(element::Type_t::i32, shape);
    copy_data(b, vector<int32_t>{0x40000140, 0x40000001, 0x40000005, 0});
    auto result = backend->create_tensor(element::Type_t::boolean, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ((vector<char>{0, 0, 1, 1}), read_vector<char>(result)); // NNP result {1, 1, 0, 1}
}

NGRAPH_TEST(${BACKEND_NAME}, lesseq_bool)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Type_t::boolean, shape);
    auto B = make_shared<op::Parameter>(element::Type_t::boolean, shape);
    auto f = make_shared<Function>(make_shared<op::v1::LessEqual>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::boolean, shape);
    copy_data(a, vector<char>{1, 1, 1, 1, 1, 1, 1, 1});
    auto b = backend->create_tensor(element::Type_t::boolean, shape);
    copy_data(b, vector<char>{0, 0, 0, 0, 0, 0, 0, 0});
    auto result = backend->create_tensor(element::Type_t::boolean, shape);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the
    // right value.
    copy_data(result, vector<char>{1, 1, 1, 1, 1, 1, 1, 1});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ((vector<char>{0, 0, 0, 0, 0, 0, 0, 0}), read_vector<char>(result));
}
