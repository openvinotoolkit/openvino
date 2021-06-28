// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "util/unary_test.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, negative)
{
    test::make_unary_test<TestEngine, op::Negative, element::f32>(Shape{2, 3})
        .test({1, -2, 0, -4.75f, 8.75f, -8.75f},
              {-1, 2, 0, 4.75f, -8.75f, 8.75f},
              MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, negative_i32)
{
    test::make_unary_test<TestEngine, op::Negative, element::i32>(Shape{2, 5})
        .test({1, 8, -8, 17, -2, 1, 8, -8, 17, -1}, {-1, -8, 8, -17, 2, -1, -8, 8, -17, 1});
}

NGRAPH_TEST(${BACKEND_NAME}, negative_f32)
{
    test::make_unary_test<TestEngine, op::Negative, element::f32>(Shape{2, 5})
        .test({1.35f, 8.76f, -8.0f, 17.234f, -2.121f, 1.0f, 8.7f, -8.92f, 17.0f, -1.0f},
              {-1.35f, -8.76f, 8.0f, -17.234f, 2.121f, -1.0f, -8.7f, 8.92f, -17.0f, 1.0f});
}
