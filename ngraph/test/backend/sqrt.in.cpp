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

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif

#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on

#include "gtest/gtest.h"
#include "runtime/backend.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, sqrt)
{
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sqrt>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, shape);
    copy_data(a, vector<float>{16, 4, 81, 100, 10000, 0});
    auto result = backend->create_tensor(element::Type_t::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(vector<float>{4, 2, 9, 10, 100, 0}, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, sqrt_negative_inputs)
{
    Shape shape{4};
    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sqrt>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, shape);
    copy_data(a, vector<float>{-1, 4, -81, 100});
    auto result = backend->create_tensor(element::Type_t::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    auto result_val = read_vector<float>(result);
    EXPECT_TRUE(isnan(result_val[0]));
    EXPECT_FLOAT_EQ(result_val[1], std::sqrt(4));
    EXPECT_TRUE(isnan(result_val[2]));
    EXPECT_FLOAT_EQ(result_val[3], std::sqrt(100));
}
