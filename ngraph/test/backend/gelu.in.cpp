// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "util/unary_test.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, gelu_erf_mode_inference_f32_8D)
{
    test::make_unary_test<TestEngine, op::v7::Gelu, element::f32>(Shape{8}).test(
        {-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0},
        {-0.00012636185,
         -0.0040495098,
         -0.04550028,
         -0.15865529,
         0.0,
         0.8413447,
         1.9544997,
         2.9959507},
        1e-4f);
}

NGRAPH_TEST(${BACKEND_NAME}, gelu_tanh_mode_inference_f32_8D)
{
    test::make_unary_test<TestEngine, op::v7::Gelu, element::f32>(Shape{8},
                                                                  op::GeluApproximationMode::TANH)
        .test({-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0},
              {-0.00012636185,
               -0.0040495098,
               -0.04550028,
               -0.15865529,
               0.0,
               0.8413447,
               1.9544997,
               2.9959507},
              1e-3f);
}

NGRAPH_TEST(${BACKEND_NAME}, gelu_erf_mode_inference_f32_3D)
{
    test::make_unary_test<TestEngine, op::v7::Gelu, element::f32>(Shape{3}).test(
        {-0.5, 0.1, 0.4}, {-0.15426877, 0.05398279, 0.2621686}, 1e-4f);
}

NGRAPH_TEST(${BACKEND_NAME}, gelu_tanh_mode_inference_f32_3D)
{
    test::make_unary_test<TestEngine, op::v7::Gelu, element::f32>(Shape{3},
                                                                  op::GeluApproximationMode::TANH)
        .test({-0.5, 0.1, 0.4}, {-0.15428599, 0.053982753, 0.262161165}, 1e-4f);
}
