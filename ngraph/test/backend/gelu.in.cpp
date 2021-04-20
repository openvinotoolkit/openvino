// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

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

NGRAPH_TEST(${BACKEND_NAME}, gelu_erf_mode_inference)
{
    Shape in_shape{3};
    element::Type et = element::f32;

    auto param = make_shared<op::Parameter>(et, in_shape);
    auto gelu = make_shared<op::v7::Gelu>(param);
    auto f = make_shared<Function>(gelu, ParameterVector{param});

    vector<float> in_vec{-0.5, 0.1, 0.4};
    vector<float> out_vec{-0.15426877,  0.05398279,  0.2621686};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input(in_shape, in_vec);
    test_case.add_expected_output(in_shape, out_vec);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, gelu_tanh_mode_inference)
{
    Shape in_shape{3};
    element::Type et = element::f32;

    auto param = make_shared<op::Parameter>(et, in_shape);
    auto gelu = make_shared<op::v7::Gelu>(param, op::GeluApproximationMode::TANH);
    auto f = make_shared<Function>(gelu, ParameterVector{param});

    vector<float> in_vec{-0.5, 0.1, 0.4};
    vector<float> out_vec{-0.15428599,  0.053982753,  0.262161165};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input(in_shape, in_vec);
    test_case.add_expected_output(in_shape, out_vec);
    test_case.run();
}
