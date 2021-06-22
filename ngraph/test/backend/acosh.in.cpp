// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backend/unary_test.hpp"

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, acosh)
{
    test_unary<element::f32>("${BACKEND_NAME}",
                             unary_func<op::Acosh>(),
                             {0.f, 1.f, -1.f, 2.f, -2.f, 3.f, -3.f, 4.f, 5.f, 10.f, 100.f},
                             std::acosh,
                             0,
                             0);
}
