// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backend/unary_test.hpp"

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, tanh_f32)
{
    test_unary<TestEngine, element::f32>(
        unary_func<op::Tanh>(), {2.f, 1.f, 0.5f, 0.f, -0.f, -0.5f, -1.f, -2.f}, std::tanh);
}

NGRAPH_TEST(${BACKEND_NAME}, tanh_int32)
{
    test_unary<TestEngine, element::i32>(
        unary_func<op::Tanh>(), {2, 1, 0, -1, -2}, {1, 1, 0, -1, -1});
}
