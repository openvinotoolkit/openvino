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
#include "ngraph/op/embedding_lookup.hpp"
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

NGRAPH_TEST(${BACKEND_NAME}, embedding_lookup_4x5_reverse)
{
    Shape shape{4};
    Shape rshape{4, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, rshape);
    auto embed = make_shared<op::EmbeddingLookup>(A, B);
    auto f0 = make_shared<Function>(NodeVector{embed}, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{3, 2, 1, 0});
    auto b = backend->create_tensor(element::f32, rshape);
    copy_data(b,
              vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto result0 = backend->create_tensor(element::f32, rshape);
    auto handle = backend->compile(f0);
    handle->call_with_validate({result0}, {a, b});
    vector<float> expected{16, 17, 18, 19, 20, 11, 12, 13, 14, 15, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5};
    EXPECT_TRUE(test::all_close_f(expected, read_vector<float>(result0), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, embedding_lookup_10x1_arbitrary)
{
    Shape shape{10};
    Shape rshape{10, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, rshape);
    auto embed = make_shared<op::EmbeddingLookup>(A, B);
    auto f0 = make_shared<Function>(NodeVector{embed}, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{9, 2, 1, 0, 3, 5, 4, 6, 8, 7});
    auto b = backend->create_tensor(element::f32, rshape);
    copy_data(b, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto result0 = backend->create_tensor(element::f32, rshape);
    auto handle = backend->compile(f0);
    handle->call_with_validate({result0}, {a, b});
    vector<float> expected{9, 2, 1, 0, 3, 5, 4, 6, 8, 7};
    EXPECT_TRUE(test::all_close_f(expected, read_vector<float>(result0), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, embedding_lookup_10x1_arbitrary_index_type_int)
{
    Shape shape{10};
    Shape rshape{10, 1};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::f32, rshape);
    auto embed = make_shared<op::EmbeddingLookup>(A, B);
    auto f0 = make_shared<Function>(NodeVector{embed}, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int>{9, 2, 1, 0, 3, 5, 4, 6, 8, 7});
    auto b = backend->create_tensor(element::f32, rshape);
    copy_data(b, vector<float>{0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5});
    auto result0 = backend->create_tensor(element::f32, rshape);
    auto handle = backend->compile(f0);
    handle->call_with_validate({result0}, {a, b});
    // vector<float> expected{9.5, 2.5, 1.5, 0.5, 3.5, 5.5, 4.5, 6.5, 8.5, 7.5};
    vector<float> expected{9.5, 2.5, 1.5, 0.5, 3.5, 5.5, 4.5, 6.5, 8.5, 7.5};
    EXPECT_TRUE(test::all_close_f(expected, read_vector<float>(result0), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, embedding_lookup_10x1_arbitrary_index_type_int64)
{
    Shape shape{10};
    Shape rshape{10, 1};
    auto A = make_shared<op::Parameter>(element::i64, shape);
    auto B = make_shared<op::Parameter>(element::f32, rshape);
    auto embed = make_shared<op::EmbeddingLookup>(A, B);
    auto f0 = make_shared<Function>(NodeVector{embed}, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i64, shape);
    copy_data(a, vector<int64_t>{9, 2, 1, 0, 3, 5, 4, 6, 8, 7});
    auto b = backend->create_tensor(element::f32, rshape);
    copy_data(b, vector<float>{0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5});
    auto result0 = backend->create_tensor(element::f32, rshape);
    auto handle = backend->compile(f0);
    handle->call_with_validate({result0}, {a, b});
    // vector<float> expected{9.5, 2.5, 1.5, 0.5, 3.5, 5.5, 4.5, 6.5, 8.5, 7.5};
    vector<float> expected{9.5, 2.5, 1.5, 0.5, 3.5, 5.5, 4.5, 6.5, 8.5, 7.5};
    EXPECT_TRUE(test::all_close_f(expected, read_vector<float>(result0), MIN_FLOAT_TOLERANCE_BITS));
}
