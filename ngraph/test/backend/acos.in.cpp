// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backend/unary_test.hpp"

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, acos)
{
    test_unary<element::f32>(
        "${BACKEND_NAME}",
        unary_func<op::Acos>(),
        {-1.f, -0.75f, -0.5f, -0.25f, -0.125f, 0.f, 0.125f, 0.25f, 0.5f, 0.75f, 1.f},
        {3.14159265f,
         2.41885841f,
         2.09439510f,
         1.82347658f,
         1.69612416f,
         1.57079633f,
         1.44546850f,
         1.31811607f,
         1.04719755f,
         0.72273425f,
         0.00000000f},
        0,
        0);
}
