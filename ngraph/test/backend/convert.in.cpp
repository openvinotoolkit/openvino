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

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, convert_int32_float32)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Type_t::i32, shape);
    auto f = make_shared<Function>(make_shared<op::Convert>(A, element::Type_t::f32),
                                   ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::i32, shape);
    copy_data(a, vector<int32_t>{281, 2, 3, 4});
    auto result = backend->create_tensor(element::Type_t::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{281, 2, 3, 4}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, convert_uint16_float32)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Type_t::u16, shape);
    auto f = make_shared<Function>(make_shared<op::Convert>(A, element::Type_t::f32),
                                   ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::u16, shape);
    copy_data(a, vector<uint16_t>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::Type_t::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1, 2, 3, 4}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, convert_int32_bool)
{
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::Type_t::i32, shape);
    auto f = make_shared<Function>(make_shared<op::Convert>(A, element::Type_t::boolean),
                                   ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    int32_t lowest = std::numeric_limits<int32_t>::lowest();
    int32_t max = std::numeric_limits<int32_t>::max();

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::i32, shape);
    copy_data(a, vector<int32_t>{0, 12, 23, 0, lowest, max});
    auto result = backend->create_tensor(element::Type_t::boolean, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<char>{0, 1, 1, 0, 1, 1}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, convert_float32_bool)
{
    Shape shape{3, 3};
    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Convert>(A, element::Type_t::boolean),
                                   ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    float lowest = std::numeric_limits<float>::lowest();
    float max = std::numeric_limits<float>::max();
    float min = std::numeric_limits<float>::min();
    float pos_inf = std::numeric_limits<float>::infinity();
    float neg_inf = -std::numeric_limits<float>::infinity();

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, shape);
    copy_data(a, vector<float>{0.f, 1.5745f, 0.12352f, 0.f, lowest, max, min, pos_inf, neg_inf});
    auto result = backend->create_tensor(element::Type_t::boolean, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<char>{0, 1, 1, 0, 1, 1, 1, 1, 1}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, convert_float32_bf16)
{
    Shape shape_a{1, 1, 3, 5};

    // input data
    vector<float> a_data = {
        0.5f, 1.5f, 0.5f, 2.5f, 1.5f, 0.5f, 3.5f, 2.5f, 0.5f, 0.5f, 2.5f, 0.5f, 0.5f, 0.5f, 1.5f};

    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape_a);
    auto convert = make_shared<op::Convert>(A, element::Type_t::bf16);
    auto f = make_shared<Function>(NodeVector{convert}, ParameterVector{A});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, shape_a);
    copy_data(a, a_data);
    auto result = backend->create_tensor(element::Type_t::bf16, shape_a);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<bfloat16>{
                  0.5, 1.5, 0.5, 2.5, 1.5, 0.5, 3.5, 2.5, 0.5, 0.5, 2.5, 0.5, 0.5, 0.5, 1.5}),
              read_vector<bfloat16>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, convert_bf16_float32)
{
    Shape shape_a{1, 1, 3, 5};

    // input data
    vector<bfloat16> a_data = {
        0.5, 1.5, 0.5, 2.5, 1.5, 0.5, 3.5, 2.5, 0.5, 0.5, 2.5, 0.5, 0.5, 0.5, 1.5};

    auto A = make_shared<op::Parameter>(element::Type_t::bf16, shape_a);
    auto convert = make_shared<op::Convert>(A, element::Type_t::f32);
    auto f = make_shared<Function>(NodeVector{convert}, ParameterVector{A});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::bf16, shape_a);
    copy_data(a, a_data);
    auto result = backend->create_tensor(element::Type_t::f32, shape_a);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<float>{0.5f,
                             1.5f,
                             0.5f,
                             2.5f,
                             1.5f,
                             0.5f,
                             3.5f,
                             2.5f,
                             0.5f,
                             0.5f,
                             2.5f,
                             0.5f,
                             0.5f,
                             0.5f,
                             1.5f}),
              read_vector<float>(result));
}
