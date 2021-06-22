// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backend/unary_test.hpp"

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, tan)
{
    test_unary<element::f32>("${BACKEND_NAME}",
                             unary_func<op::Tan>(),
                             {0.f, 0.25f, -0.25f, 0.5f, -0.5f, 1.f, -1.f, 2.f, -2.f, 4.f, -4.f},
                             {0.00000000f,
                              0.25534192f,
                              -0.25534192f,
                              0.54630249f,
                              -0.54630249f,
                              1.55740772f,
                              -1.55740772f,
                              -2.18503986f,
                              2.18503986f,
                              1.15782128f,
                              -1.15782128f},
                             0,
                             0);
}
