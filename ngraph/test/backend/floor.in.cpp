// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backend/unary_test.hpp"

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, floor)
{
    test_unary<element::f32>("${BACKEND_NAME}",
                             unary_func<op::Floor>(),
                             {-2.5f, -2.0f, 0.3f, 4.8f},
                             {-3.0f, -2.0f, 0.0f, 4.0f},
                             {2, 2},
                             {2, 2},
                             0,
                             0);
}

NGRAPH_TEST(${BACKEND_NAME}, floor_int32)
{
    test_unary<element::i32>("${BACKEND_NAME}",
                             unary_func<op::Floor>(),
                             {-2, -136314888, 0x40000010, 0x40000001},
                             {-2, -136314888, 0x40000010, 0x40000001},
                             {2, 2},
                             {2, 2},
                             0,
                             0);
}

NGRAPH_TEST(${BACKEND_NAME}, floor_int64)
{
    // This tests large numbers that will not fit in a double
    test_unary<element::i64>("${BACKEND_NAME}",
                             unary_func<op::Floor>(),
                             {0, 1, 0x4000000000000001},
                             {0, 1, 0x4000000000000001},
                             0,
                             0);
}
