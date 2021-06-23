// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "backend/unary_test.hpp"

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, asinh)
{
    test_unary<TestEngine, element::f32>(
        unary_func<op::Asinh>(),
        {0.f, 1.f, -1.f, 2.f, -2.f, 3.f, -3.f, 4.f, 5.f, 10.f, 100.f},
        std::asinh);
}
