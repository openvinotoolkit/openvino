// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backend/unary_test.hpp"

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, negative)
{
    test_unary<TestEngine, element::f32>(unary_func<op::Negative>(),
                                         {1, -2, 0, -4.75f, 8.75f, -8.75f},
                                         {-1, 2, 0, 4.75f, -8.75f, 8.75f},
                                         Shape{2, 3},
                                         MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, negative_i32)
{
    test_unary<TestEngine, element::i32>(unary_func<op::Negative>(),
                                         {1, 8, -8, 17, -2, 1, 8, -8, 17, -1},
                                         {-1, -8, 8, -17, 2, -1, -8, 8, -17, 1},
                                         Shape{2, 5});
}

NGRAPH_TEST(${BACKEND_NAME}, negative_f32)
{
    test_unary<TestEngine, element::f32>(
        unary_func<op::Negative>(),
        {1.35f, 8.76f, -8.0f, 17.234f, -2.121f, 1.0f, 8.7f, -8.92f, 17.0f, -1.0f},
        {-1.35f, -8.76f, 8.0f, -17.234f, 2.121f, -1.0f, -8.7f, 8.92f, -17.0f, 1.0f},
        Shape{2, 5});
}
