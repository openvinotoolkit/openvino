// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backend/unary_test.hpp"

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, ceiling)
{
    test_unary<element::f32>("${BACKEND_NAME}",
                             unary_func<op::Ceiling>(),
                             {-2.5f, -2.0f, 0.3f, 4.8f},
                             {-2.0f, -2.0f, 1.0f, 5.0f},
                             {2, 2},
                             {2, 2},
                             0,
                             0);
}

NGRAPH_TEST(${BACKEND_NAME}, ceiling_int64)
{
    test_unary<element::i64>("${BACKEND_NAME}",
                             unary_func<op::Ceiling>(),
                             {0, 1, 0x4000000000000001},
                             {0, 1, 0x4000000000000001},
                             0,
                             0);
}
