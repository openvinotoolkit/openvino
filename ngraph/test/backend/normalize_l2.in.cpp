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
#include <numeric>
#include <random>
#include <string>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

// ----------------------- eps_mode = ngraph::op::EpsMode::ADD ----------------------- //

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_all_mode_add)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axes = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{0, 1});
    float eps = 1e-7;
    auto f = make_shared<Function>(
        make_shared<op::v0::NormalizeL2>(A, axes, eps, ngraph::op::EpsMode::ADD),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{0.18257418, 0.36514837, 0.5477226, 0.73029673}),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_none_mode_add)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axes = make_shared<op::Constant>(element::i64, Shape{0}, vector<int64_t>{});
    float eps = 1e-7;
    auto f = make_shared<Function>(
        make_shared<op::v0::NormalizeL2>(A, axes, eps, ngraph::op::EpsMode::ADD),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{0.18257418, 0.36514837, 0.5477226, 0.73029673}),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_zero_mode_add)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axes = make_shared<op::Constant>(element::i64, Shape{}, vector<int64_t>{0});
    float eps = 1e-7;
    auto f = make_shared<Function>(
        make_shared<op::v0::NormalizeL2>(A, axes, eps, ngraph::op::EpsMode::ADD),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{0.31622776, 0.4472136, 0.94868326, 0.8944272}),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_one_mode_add)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axes = make_shared<op::Constant>(element::i64, Shape{}, vector<int64_t>{1});
    float eps = 1e-7;
    auto f = make_shared<Function>(
        make_shared<op::v0::NormalizeL2>(A, axes, eps, ngraph::op::EpsMode::ADD),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{0.4472136, 0.8944272, 0.6, 0.8}),
                                  read_vector<float>(result)));
}

// ----------------------- eps_mode = ngraph::op::EpsMode::MAX ----------------------- //

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_all_mode_max)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axes = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{0, 1});
    float eps = 1e-7;
    auto f = make_shared<Function>(
        make_shared<op::v0::NormalizeL2>(A, axes, eps, ngraph::op::EpsMode::ADD),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{0.18257419, 0.36514837, 0.54772256, 0.73029674}),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_none_mode_max)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axes = make_shared<op::Constant>(element::i64, Shape{0}, vector<int64_t>{});
    float eps = 1e-7;
    auto f = make_shared<Function>(
        make_shared<op::v0::NormalizeL2>(A, axes, eps, ngraph::op::EpsMode::MAX),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{0.18257419, 0.36514837, 0.54772256, 0.7302967}),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_zero_mode_max)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axes = make_shared<op::Constant>(element::i64, Shape{}, vector<int64_t>{0});
    float eps = 1e-7;
    auto f = make_shared<Function>(
        make_shared<op::v0::NormalizeL2>(A, axes, eps, ngraph::op::EpsMode::MAX),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{0.31622777, 0.4472136, 0.9486833, 0.89442719}),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_one_mode_max)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axes = make_shared<op::Constant>(element::i64, Shape{}, vector<int64_t>{1});
    float eps = 1e-7;
    auto f = make_shared<Function>(
        make_shared<op::v0::NormalizeL2>(A, axes, eps, ngraph::op::EpsMode::MAX),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{0.4472136, 0.89442719, 0.6, 0.8}),
                                  read_vector<float>(result)));
}
