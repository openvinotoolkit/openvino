// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backend/unary_test.hpp"

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

template <typename T>
static T f_sigmoid(T x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

NGRAPH_TEST(${BACKEND_NAME}, sigmoid_n1c1h2w2)
{
    test_unary<TestEngine, element::f32>(
        unary_func<op::Sigmoid>(), {1.0f, 4.0f, 1.0f, 4.0f}, f_sigmoid, Shape{1, 1, 2, 2});
}

NGRAPH_TEST(${BACKEND_NAME}, sigmoid_n1c1h4)
{
    test_unary<TestEngine, element::f32>(
        unary_func<op::Sigmoid>(), {1.0f, 4.0f, 1.0f, 4.0f}, f_sigmoid, Shape{1, 1, 4});
}
