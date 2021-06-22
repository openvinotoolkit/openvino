// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backend/unary_test.hpp"

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, erf)
{
    test_unary<element::f32>("${BACKEND_NAME}",
                             unary_func<op::Erf>(),
                             {-4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f},
                             std::erf,
                             0,
                             0);
}
