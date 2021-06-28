// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "util/unary_test.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, atan)
{
    auto t = test::make_unary_test<TestEngine, op::Atan, element::f32>(Shape{11});

    t.test({-4.f, -2.f, -1.f, -0.5f, -0.25f, 0.f, 0.25f, 0.5f, 1.f, 2.f, 4.f},
           {-1.32581766f,
            -1.10714872f,
            -0.78539816f,
            -0.46364761f,
            -0.24497866f,
            0.00000000f,
            0.24497866f,
            0.46364761f,
            0.78539816f,
            1.10714872f,
            1.32581766f});
}
