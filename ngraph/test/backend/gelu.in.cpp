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
#include "util/random.hpp"

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

NGRAPH_TEST(${BACKEND_NAME}, gelu_f32)
{
    Shape shape{100000};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Gelu>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-100.0f, 100.0f);
    vector<vector<float>> args;
    for (shared_ptr<op::Parameter> param : f->get_parameters())
    {
        auto name = param->get_name();
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, args[0]);
    auto result = backend->create_tensor(element::f32, shape);

    std::transform(args[0].begin(), args[0].end(), args[0].begin(), [](float x) -> float {
        return 0.5f * x * (1.0f + erf(x / sqrt(2.0f)));
    });

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close(args[0], read_vector<float>(result), .007f, .007f));
}

NGRAPH_TEST(${BACKEND_NAME}, gelu_f64)
{
    Shape shape{8};
    auto A = make_shared<op::Parameter>(element::f64, shape);
    auto f = make_shared<Function>(make_shared<op::Gelu>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f64, shape);
    vector<double> input{-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0};
    copy_data(a, input);
    auto result = backend->create_tensor(element::f64, shape);

    std::transform(input.begin(), input.end(), input.begin(), [](double x) -> double {
        return 0.5 * x * (1.0 + erf(x / sqrt(2.0)));
    });

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(input, read_vector<double>(result)));
}

static double gelu_backprop_factor(double x)
{
    auto pi = 4.0 * std::atan(1.0);
    return 0.5 * (1.0 + erf(x * sqrt(1.0 / 2.0))) + (x * exp(-x * x / 2.0)) / sqrt(2.0 * pi);
}

NGRAPH_TEST(${BACKEND_NAME}, gelu_backprop_factor_f32)
{
    Shape shape{8};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::GeluBackpropFactor>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> input{-4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    copy_data(a, input);
    auto result = backend->create_tensor(element::f32, shape);

    std::transform(input.begin(), input.end(), input.begin(), [](float x) -> float {
        return static_cast<float>(gelu_backprop_factor(static_cast<double>(x)));
    });

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(
        test::all_close_f(input, read_vector<float>(result), DEFAULT_FLOAT_TOLERANCE_BITS + 6));
}

NGRAPH_TEST(${BACKEND_NAME}, gelu_backprop_factor_f64)
{
    Shape shape{8};
    auto A = make_shared<op::Parameter>(element::f64, shape);
    auto f = make_shared<Function>(make_shared<op::GeluBackpropFactor>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f64, shape);
    vector<double> input{-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0};
    copy_data(a, input);
    auto result = backend->create_tensor(element::f64, shape);

    std::transform(input.begin(), input.end(), input.begin(), [](double x) -> double {
        return gelu_backprop_factor(x);
    });

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(input, read_vector<double>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_gelu_f32)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape{8};
    auto make_graph = [shape]() {
        auto A = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Gelu>(A), ParameterVector{A});
    };

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> input{-4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    copy_data(a, input);

    EXPECT_TRUE(autodiff_numeric_compare<float>(backend.get(), make_graph, {a}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_gelu_f64)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape{8};
    auto make_graph = [shape]() {
        auto A = make_shared<op::Parameter>(element::f64, shape);
        return make_shared<Function>(make_shared<op::Gelu>(A), ParameterVector{A});
    };

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f64, shape);
    vector<double> input{-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0};
    copy_data(a, input);

    EXPECT_TRUE(autodiff_numeric_compare<double>(backend.get(), make_graph, {a}, .01, .01));
}
