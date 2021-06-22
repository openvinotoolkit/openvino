// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backend/unary_test.hpp"

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, sign)
{
    test_unary<element::f32>("${BACKEND_NAME}",
                             unary_func<op::Sign>(),
                             {1, -2, 0, -4.8f, 4.8f, -0.0f},
                             {1, -1, 0, -1, 1, 0},
                             Shape{2, 3},
                             Shape{2, 3},
                             0,
                             0);
}
