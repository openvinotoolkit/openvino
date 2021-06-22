// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backend/unary_test.hpp"

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, abs)
{
    test_unary<element::f32>("${BACKEND_NAME}",
                             unary_func<op::Abs>(),
                             {1, -2, 0, -4.75f},
                             {1, 2, 0, 4.75f},
                             {2, 2},
                             {2, 2},
                             0,
                             0);
}
