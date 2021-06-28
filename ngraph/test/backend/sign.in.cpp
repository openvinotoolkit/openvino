// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "util/unary_test.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, sign)
{
    test::make_unary_test<TestEngine, op::Sign, element::f32>(Shape{2, 3})
        .test({1, -2, 0, -4.8f, 4.8f, -0.0f}, {1, -1, 0, -1, 1, 0}, MIN_FLOAT_TOLERANCE_BITS);
}
