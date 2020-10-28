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

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, one_hot_scalar_2_in_3)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    int axis = 0;
    Shape shape_r{3};
    auto depth = op::Constant::create(element::i32, {}, {shape_r[axis]});
    auto on_value = op::Constant::create(element::i32, {}, {1});
    auto off_value = op::Constant::create(element::i32, {}, {0});
    auto r = make_shared<op::v1::OneHot>(A, depth, on_value, off_value, axis);
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{2});
    auto result = backend->create_tensor(element::i32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int32_t>{0, 0, 1}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_scalar_1_in_3)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    int axis = 0;
    Shape shape_r{3};
    auto depth = op::Constant::create(element::i32, {}, {shape_r[axis]});
    auto on_value = op::Constant::create(element::i32, {}, {1});
    auto off_value = op::Constant::create(element::i32, {}, {0});
    auto r = make_shared<op::v1::OneHot>(A, depth, on_value, off_value, axis);
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{1});
    auto result = backend->create_tensor(element::i32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int32_t>{0, 1, 0}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_scalar_0_in_3)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_r{3};
    int axis = 0;
    auto depth = op::Constant::create(element::i32, {}, {shape_r[axis]});
    auto on_value = op::Constant::create(element::i32, {}, {1});
    auto off_value = op::Constant::create(element::i32, {}, {0});
    auto r = make_shared<op::v1::OneHot>(A, depth, on_value, off_value, axis);
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{0});
    auto result = backend->create_tensor(element::i32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int32_t>{1, 0, 0}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_vector_0)
{
    Shape shape_a{8};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_r{3, 8};
    int axis = 0;
    auto depth = op::Constant::create(element::i32, {}, {shape_r[axis]});
    auto on_value = op::Constant::create(element::i32, {}, {1});
    auto off_value = op::Constant::create(element::i32, {}, {0});
    auto r = make_shared<op::v1::OneHot>(A, depth, on_value, off_value, axis);
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{2, 1, 0, 0, 2, 2, 1, 0});
    auto result = backend->create_tensor(element::i32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ(
        (vector<int32_t>{0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0}),
        read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_vector_1)
{
    Shape shape_a{8};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_r{8, 3};
    int axis = 1;
    auto depth = op::Constant::create(element::i32, {}, {shape_r[axis]});
    auto on_value = op::Constant::create(element::i32, {}, {1});
    auto off_value = op::Constant::create(element::i32, {}, {0});
    auto r = make_shared<op::v1::OneHot>(A, depth, on_value, off_value, axis);
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{2, 1, 0, 0, 2, 2, 1, 0});
    auto result = backend->create_tensor(element::i32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ(
        (vector<int32_t>{0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0}),
        read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_vector_1_barely_oob)
{
    Shape shape_a{8};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_r{8, 3};
    int axis = 1;
    auto depth = op::Constant::create(element::i32, {}, {shape_r[axis]});
    auto on_value = op::Constant::create(element::i32, {}, {1});
    auto off_value = op::Constant::create(element::i32, {}, {0});
    auto r = make_shared<op::v1::OneHot>(A, depth, on_value, off_value, axis);
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{2, 1, 0, 0, 3, 2, 1, 0});
    auto result = backend->create_tensor(element::i32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    vector<int32_t> rv = read_vector<int32_t>(result);

    EXPECT_EQ(rv[0], 0);
    EXPECT_EQ(rv[1], 0);
    EXPECT_EQ(rv[2], 1);

    EXPECT_EQ(rv[3], 0);
    EXPECT_EQ(rv[4], 1);
    EXPECT_EQ(rv[5], 0);

    EXPECT_EQ(rv[6], 1);
    EXPECT_EQ(rv[7], 0);
    EXPECT_EQ(rv[8], 0);

    EXPECT_EQ(rv[9], 1);
    EXPECT_EQ(rv[10], 0);
    EXPECT_EQ(rv[11], 0);

    // These are undefined since value is out of bounds
    // EXPECT_EQ(rv[12], 0);
    // EXPECT_EQ(rv[13], 0);
    // EXPECT_EQ(rv[14], 0);

    EXPECT_EQ(rv[15], 0);
    EXPECT_EQ(rv[16], 0);
    EXPECT_EQ(rv[17], 1);

    EXPECT_EQ(rv[18], 0);
    EXPECT_EQ(rv[19], 1);
    EXPECT_EQ(rv[20], 0);

    EXPECT_EQ(rv[21], 1);
    EXPECT_EQ(rv[22], 0);
    EXPECT_EQ(rv[23], 0);
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_matrix_0)
{
    Shape shape_a{3, 3};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_r{3, 3, 3};
    int axis = 0;
    auto depth = op::Constant::create(element::i32, {}, {shape_r[axis]});
    auto on_value = op::Constant::create(element::i32, {}, {1});
    auto off_value = op::Constant::create(element::i32, {}, {0});
    auto r = make_shared<op::v1::OneHot>(A, depth, on_value, off_value, axis);
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a,
              vector<int32_t>{
                  0, 1, 1, 2, 1, 0, 0, 2, 1,
              });
    auto result = backend->create_tensor(element::i32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int32_t>{1, 0, 0, 0, 0, 1, 1, 0, 0,

                               0, 1, 1, 0, 1, 0, 0, 0, 1,

                               0, 0, 0, 1, 0, 0, 0, 1, 0}),
              read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_vector_many_categories)
{
    // Imagenet has roughly 20,000 categories
    uint32_t category_count = 20000;
    Shape shape_a{6};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_r{6, category_count};
    int axis = 1;
    auto depth = op::Constant::create(element::i32, {}, {shape_r[axis]});
    auto on_value = op::Constant::create(element::i32, {}, {1});
    auto off_value = op::Constant::create(element::i32, {}, {0});
    auto r = make_shared<op::v1::OneHot>(A, depth, on_value, off_value, axis);
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    vector<int32_t> input_data{0, 11, 101, 1001, 10001, static_cast<int32_t>(category_count - 1)};
    copy_data(a, input_data);
    auto result = backend->create_tensor(element::i32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    vector<int32_t> data = read_vector<int32_t>(result);

    vector<int32_t> bit_positions;
    for (size_t i = 0; i < shape_size(shape_r); ++i)
    {
        if (data[i] == 1)
        {
            bit_positions.push_back(i % category_count);
        }
    }
    EXPECT_EQ(bit_positions, input_data);
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_on_off_float)
{
    Shape shape_a{3, 3};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_r{3, 3, 3};
    int axis = 0;
    auto depth = op::Constant::create(element::i32, {}, {shape_r[axis]});
    auto on_value = op::Constant::create(element::f32, {}, {2.5});
    auto off_value = op::Constant::create(element::f32, {}, {0.5});
    auto r = make_shared<op::v1::OneHot>(A, depth, on_value, off_value, axis);
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a,
              vector<int32_t>{
                  0, 1, 1, 2, 1, 0, 0, 2, 1,
              });
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<float>{2.5, 0.5, 0.5, 0.5, 0.5, 2.5, 2.5, 0.5, 0.5, 0.5, 2.5, 2.5, 0.5, 2.5,
                             0.5, 0.5, 0.5, 2.5, 0.5, 0.5, 0.5, 2.5, 0.5, 0.5, 0.5, 2.5, 0.5}),
              read_vector<float>(result));
}
