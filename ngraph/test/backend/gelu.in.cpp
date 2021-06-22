// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backend/unary_test.hpp"

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, gelu_erf_mode_inference_f32_8D)
{
    test_unary<element::f32>("${BACKEND_NAME}",
                             unary_func<op::v7::Gelu>(),
                             {-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0},
                             {-0.00012636185,
                              -0.0040495098,
                              -0.04550028,
                              -0.15865529,
                              0.0,
                              0.8413447,
                              1.9544997,
                              2.9959507},
                             1e-4,
                             0);
}

NGRAPH_TEST(${BACKEND_NAME}, gelu_tanh_mode_inference_f32_8D)
{
    test_unary<element::f32>("${BACKEND_NAME}",
                             unary_func<op::v7::Gelu>(op::GeluApproximationMode::TANH),
                             {-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0},
                             {-0.00012636185,
                              -0.0040495098,
                              -0.04550028,
                              -0.15865529,
                              0.0,
                              0.8413447,
                              1.9544997,
                              2.9959507},
                             1e-3f,
                             0);
}

NGRAPH_TEST(${BACKEND_NAME}, gelu_erf_mode_inference_f32_3D)
{
    test_unary<element::f32>("${BACKEND_NAME}",
                             unary_func<op::v7::Gelu>(),
                             {-0.5, 0.1, 0.4},
                             {-0.15426877, 0.05398279, 0.2621686},
                             1e-4f,
                             0);
}

NGRAPH_TEST(${BACKEND_NAME}, gelu_tanh_mode_inference_f32_3D)
{
    test_unary<element::f32>("${BACKEND_NAME}",
                             unary_func<op::v7::Gelu>(op::GeluApproximationMode::TANH),
                             {-0.5, 0.1, 0.4},
                             {-0.15428599, 0.053982753, 0.262161165},
                             1e-4f,
                             0);
}
