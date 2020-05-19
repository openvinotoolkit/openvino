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
#include "util/autodiff/numeric_compare.hpp"
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

NGRAPH_TEST(${BACKEND_NAME}, layer_norm_bprop_affine_stats)
{
    auto p_data = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto p_delta = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto p_mean = make_shared<op::Parameter>(element::f32, Shape{2});
    auto p_var = make_shared<op::Parameter>(element::f32, Shape{2});
    auto p_scale = make_shared<op::Parameter>(element::f32, Shape{4});
    auto lnb = make_shared<op::LayerNormBackprop>(p_data, p_delta, p_mean, p_var, p_scale);
    auto f = make_shared<Function>(lnb->outputs(),
                                   ParameterVector{p_data, p_delta, p_mean, p_var, p_scale});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create tensors for input
    auto data = backend->create_tensor(element::f32, Shape{2, 4});
    auto delta = backend->create_tensor(element::f32, Shape{2, 4});
    auto mean = backend->create_tensor(element::f32, Shape{2});
    auto var = backend->create_tensor(element::f32, Shape{2});
    auto scale = backend->create_tensor(element::f32, Shape{4});
    // Fill in input tensors
    vector<float> d_input{-4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    copy_data(data, d_input);
    vector<float> dt_input{0.1f, -0.1f, 0.2f, -0.2f, 0.1f, -0.1f, 0.2f, -0.2f};
    copy_data(delta, dt_input);
    vector<float> s_input{-1.0f, 1.0f, 2.0f, 3.0f};
    copy_data(scale, s_input);
    vector<float> m_input{-2.5f, 1.5f};
    copy_data(mean, m_input);
    vector<float> v_input{1.25f, 1.25f};
    copy_data(var, v_input);
    // Create tensors for output
    auto d_data = backend->create_tensor(element::f32, Shape{2, 4});
    auto d_scale = backend->create_tensor(element::f32, Shape{4});
    auto d_bias = backend->create_tensor(element::f32, Shape{4});

    // Expected results (Manually compute)
    vector<float> exp_d_data{-0.1341624855995178223f,
                             -0.04472083225846290588f,
                             0.4919326305389404297f,
                             -0.31304931640625f,
                             -0.1341624855995178223f,
                             -0.04472083225846290588f,
                             0.4919326305389404297f,
                             -0.31304931640625f};
    vector<float> exp_d_scale{-0.2683270871639251709f,
                              0.08944236487150192261f,
                              0.1788847297430038452f,
                              -0.5366541743278503418f};
    vector<float> exp_d_bias{0.2f, -0.2f, 0.4f, -0.4f};

    auto handle = backend->compile(f);
    handle->call_with_validate({d_data, d_scale, d_bias}, {data, delta, mean, var, scale});
    EXPECT_TRUE(test::all_close_f(exp_d_data, read_vector<float>(d_data)));
    EXPECT_TRUE(test::all_close_f(exp_d_scale, read_vector<float>(d_scale)));
    EXPECT_TRUE(test::all_close_f(exp_d_bias, read_vector<float>(d_bias)));
}

NGRAPH_TEST(${BACKEND_NAME}, layer_norm_bprop_affine)
{
    auto p_data = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto p_delta = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto p_scale = make_shared<op::Parameter>(element::f32, Shape{4});
    auto lnb = make_shared<op::LayerNormBackprop>(p_data, p_delta, p_scale);
    auto f = make_shared<Function>(lnb->outputs(), ParameterVector{p_data, p_delta, p_scale});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create tensors for input
    auto data = backend->create_tensor(element::f32, Shape{2, 4});
    auto delta = backend->create_tensor(element::f32, Shape{2, 4});
    auto scale = backend->create_tensor(element::f32, Shape{4});
    // Fill in input tensors
    vector<float> d_input{-4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    copy_data(data, d_input);
    vector<float> dt_input{0.1f, -0.1f, 0.2f, -0.2f, 0.1f, -0.1f, 0.2f, -0.2f};
    copy_data(delta, dt_input);
    vector<float> s_input{-1.0f, 1.0f, 2.0f, 3.0f};
    copy_data(scale, s_input);
    // Create tensors for output
    auto d_data = backend->create_tensor(element::f32, Shape{2, 4});
    auto d_scale = backend->create_tensor(element::f32, Shape{4});
    auto d_bias = backend->create_tensor(element::f32, Shape{4});

    // Expected results (Manually computed)
    vector<float> exp_d_data{-0.1341624855995178223f,
                             -0.04472083225846290588f,
                             0.4919326305389404297f,
                             -0.31304931640625f,
                             -0.1341624855995178223f,
                             -0.04472083225846290588f,
                             0.4919326305389404297f,
                             -0.31304931640625f};
    vector<float> exp_d_scale{-0.2683270871639251709f,
                              0.08944236487150192261f,
                              0.1788847297430038452f,
                              -0.5366541743278503418f};
    vector<float> exp_d_bias{0.2f, -0.2f, 0.4f, -0.4f};

    auto handle = backend->compile(f);
    handle->call_with_validate({d_data, d_scale, d_bias}, {data, delta, scale});
    EXPECT_TRUE(test::all_close_f(exp_d_data, read_vector<float>(d_data)));
    EXPECT_TRUE(test::all_close_f(exp_d_scale, read_vector<float>(d_scale)));
    EXPECT_TRUE(test::all_close_f(exp_d_bias, read_vector<float>(d_bias)));
}

NGRAPH_TEST(${BACKEND_NAME}, layer_norm_bprop_4d_input)
{
    auto p_data = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5});
    auto p_delta = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5});
    auto p_mean = make_shared<op::Parameter>(element::f32, Shape{2});
    auto p_variance = make_shared<op::Parameter>(element::f32, Shape{2});
    auto p_scale = make_shared<op::Parameter>(element::f32, Shape{60});
    auto lnb = make_shared<op::LayerNormBackprop>(p_data, p_delta, p_mean, p_variance, p_scale);

    auto output_data = lnb->output(0);
    auto output_scale = lnb->output(1);
    auto output_bias = lnb->output(2);

    // flatten output_scale
    auto output_scale_shape = output_scale.get_shape();
    auto flattened_output_scale = make_shared<op::Reshape>(
        output_scale, get_default_order(output_scale_shape), Shape{shape_size(output_scale_shape)});

    auto f = make_shared<Function>(OutputVector{output_data, flattened_output_scale, output_bias},
                                   ParameterVector{p_data, p_delta, p_mean, p_variance, p_scale});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create tensors for input
    auto data = backend->create_tensor(element::f32, Shape{2, 3, 4, 5});
    auto delta = backend->create_tensor(element::f32, Shape{2, 3, 4, 5});
    auto mean = backend->create_tensor(element::f32, Shape{2});
    auto variance = backend->create_tensor(element::f32, Shape{2});
    auto scale = backend->create_tensor(element::f32, Shape{60});
    // Fill in input tensors
    vector<float> d_input(2 * 3 * 4 * 5, 1);
    copy_data(data, d_input);
    vector<float> dt_input(2 * 3 * 4 * 5, 1);
    copy_data(delta, dt_input);
    vector<float> m_input(2, 1);
    copy_data(mean, m_input);
    vector<float> v_input(2, 1);
    copy_data(variance, v_input);
    vector<float> s_input(60, 1);
    copy_data(scale, s_input);
    // Create tensors for output
    auto d_data = backend->create_tensor(element::f32, Shape{2, 3, 4, 5});
    auto d_scale = backend->create_tensor(element::f32, Shape{60});
    auto d_bias = backend->create_tensor(element::f32, Shape{60});

    auto handle = backend->compile(f);
    handle->call_with_validate({d_data, d_scale, d_bias}, {data, delta, mean, variance, scale});

    vector<float> expected_data(120, 0);
    vector<float> expected_scale(60, 0);
    vector<float> expected_bias(60, 2);

    EXPECT_TRUE(test::all_close(expected_data, read_vector<float>(d_data), 1e-5f, 1e-6f));
    EXPECT_TRUE(test::all_close(expected_scale, read_vector<float>(d_scale), 1e-5f, 1e-6f));
    EXPECT_TRUE(test::all_close(expected_bias, read_vector<float>(d_bias), 1e-5f, 1e-6f));
}
