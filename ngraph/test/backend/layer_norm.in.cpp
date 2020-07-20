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

NGRAPH_TEST(${BACKEND_NAME}, layer_norm_affine_stats)
{
    auto p_data = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto p_scale = make_shared<op::Parameter>(element::f32, Shape{4});
    auto p_bias = make_shared<op::Parameter>(element::f32, Shape{4});
    auto ln = make_shared<op::LayerNorm>(p_data, p_scale, p_bias);
    auto f = make_shared<Function>(ln->outputs(), ParameterVector{p_data, p_scale, p_bias});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create tensors for input
    auto data = backend->create_tensor(element::f32, Shape{2, 4});
    auto scale = backend->create_tensor(element::f32, Shape{4});
    auto bias = backend->create_tensor(element::f32, Shape{4});
    // Fill in input tensors
    vector<float> d_input{-4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    copy_data(data, d_input);
    vector<float> s_input{-1.0f, 1.0f, 2.0f, 3.0f};
    copy_data(scale, s_input);
    vector<float> b_input{-4.0f, -3.0f, -2.0f, -1.0f};
    copy_data(bias, b_input);
    // Create tensors for output
    auto norm = backend->create_tensor(element::f32, Shape{2, 4});
    auto mean = backend->create_tensor(element::f32, Shape{2});
    auto var = backend->create_tensor(element::f32, Shape{2});

    // Expected results (Manually computed)
    vector<float> exp_norm{-2.658364534378051758f,
                           -3.447211742401123047f,
                           -1.105576276779174805f,
                           3.024906158447265625f,
                           -2.658364534378051758f,
                           -3.447211742401123047f,
                           -1.105576276779174805f,
                           3.024906158447265625f};
    vector<float> exp_mean{-2.5f, 1.5f};
    vector<float> exp_var{1.25f, 1.25f};

    auto handle = backend->compile(f);
    handle->call_with_validate({norm, mean, var}, {data, scale, bias});
    EXPECT_TRUE(test::all_close_f(exp_norm, read_vector<float>(norm)));
    EXPECT_TRUE(test::all_close_f(exp_mean, read_vector<float>(mean)));
    EXPECT_TRUE(test::all_close_f(exp_var, read_vector<float>(var)));
}
