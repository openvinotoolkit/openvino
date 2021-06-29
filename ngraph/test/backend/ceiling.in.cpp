// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "util/unary_test.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, ceiling)
{
    test::make_unary_test<TestEngine, op::Ceiling, element::f32>(Shape{2, 2})
        .test({-2.5f, -2.0f, 0.3f, 4.8f}, {-2.0f, -2.0f, 1.0f, 5.0f}, MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, ceiling_int64)
{
    test::make_unary_test<TestEngine, op::Ceiling, element::i64>(Shape{3}).test(
        {0, 1, 0x4000000000000001}, {0, 1, 0x4000000000000001}, MIN_FLOAT_TOLERANCE_BITS);
}
