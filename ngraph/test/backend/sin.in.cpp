// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backend/unary_test.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, sin)
{
    test_unary<TestEngine, element::f32>(
        unary_func<op::Sin>(),
        {0.f, 0.25f, -0.25f, 0.5f, -0.5f, 1.f, -1.f, 2.f, -2.f, 4.f, -4.f},
        {0.00000000f,
         0.24740396f,
         -0.24740396f,
         0.47942554f,
         -0.47942554f,
         0.84147098f,
         -0.84147098f,
         0.90929743f,
         -0.90929743f,
         -0.75680250f,
         0.75680250f},
        Shape{11});
}
