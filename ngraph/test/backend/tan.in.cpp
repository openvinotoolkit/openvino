// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "util/unary_test.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, tan)
{
    test::make_unary_test<TestEngine, op::Tan, element::f32>(Shape{11}).test(
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
         -1.15782128f});
}
