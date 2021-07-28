// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, gelu_erf_mode_inference_f32_8D)
{
    Shape in_shape{8};
    element::Type et = element::f32;

    auto param = make_shared<op::Parameter>(et, in_shape);
    auto gelu = make_shared<op::v7::Gelu>(param);
    auto f = make_shared<Function>(gelu, ParameterVector{param});

    vector<float> in_vec{-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0};
    vector<float> out_vec{-0.00012636185, -0.0040495098, -0.04550028, -0.15865529, 0.0, 0.8413447, 1.9544997, 2.9959507};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(in_shape, in_vec);
    test_case.add_expected_output<float>(in_shape, out_vec);
    test_case.run_with_tolerance_as_fp(1e-4f);
}

NGRAPH_TEST(${BACKEND_NAME}, gelu_tanh_mode_inference_f32_8D)
{
    Shape in_shape{8};
    element::Type et = element::f32;

    auto param = make_shared<op::Parameter>(et, in_shape);
    auto gelu = make_shared<op::v7::Gelu>(param, op::GeluApproximationMode::TANH);
    auto f = make_shared<Function>(gelu, ParameterVector{param});

    vector<float> in_vec{-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0};
    vector<float> out_vec{-0.00012636185, -0.0040495098, -0.04550028, -0.15865529, 0.0, 0.8413447, 1.9544997, 2.9959507};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(in_shape, in_vec);
    test_case.add_expected_output<float>(in_shape, out_vec);
    test_case.run_with_tolerance_as_fp(1e-3f);
}

NGRAPH_TEST(${BACKEND_NAME}, gelu_erf_mode_inference_f32_3D)
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
    test_case.run_with_tolerance_as_fp(1e-4f);
}

NGRAPH_TEST(${BACKEND_NAME}, gelu_tanh_mode_inference_f32_3D)
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
    test_case.run_with_tolerance_as_fp(1e-4f);
}
