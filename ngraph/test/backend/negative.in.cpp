// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backend/unary_test.hpp"

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, negative)
{
    test_unary<element::f32>("${BACKEND_NAME}",
                             unary_func<op::Negative>(),
                             {1, -2, 0, -4.75f, 8.75f, -8.75f},
                             {-1, 2, 0, 4.75f, -8.75f, 8.75f},
                             {2, 3},
                             {2, 3},
                             0,
                             0);
}

NGRAPH_TEST(${BACKEND_NAME}, negative_i32)
{
    test_unary<element::i32>("${BACKEND_NAME}",
                             unary_func<op::Negative>(),
                             {1, 8, -8, 17, -2, 1, 8, -8, 17, -1},
                             {-1, -8, 8, -17, 2, -1, -8, 8, -17, 1},
                             {2, 5},
                             {2, 5},
                             0,
                             0);
}

NGRAPH_TEST(${BACKEND_NAME}, negative_f32)
{
    test_unary<element::f32>(
        "${BACKEND_NAME}",
        unary_func<op::Negative>(),
        {1.35f, 8.76f, -8.0f, 17.234f, -2.121f, 1.0f, 8.7f, -8.92f, 17.0f, -1.0f},
        {-1.35f, -8.76f, 8.0f, -17.234f, 2.121f, -1.0f, -8.7f, 8.92f, -17.0f, 1.0f},
        {2, 5},
        {2, 5},
        0,
        0);
}
