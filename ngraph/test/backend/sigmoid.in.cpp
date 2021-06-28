// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "util/unary_test.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

template <typename T>
T f_sigmoid(T x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

NGRAPH_TEST(${BACKEND_NAME}, sigmoid_n1c1h2w2)
{
    test::make_unary_test<TestEngine, op::Sigmoid, element::f32>(Shape{1, 1, 2, 2})
        .test({1.0f, 4.0f, 1.0f, 4.0f}, f_sigmoid);
}

NGRAPH_TEST(${BACKEND_NAME}, sigmoid_n1c1h4)
{
    test::make_unary_test<TestEngine, op::Sigmoid, element::f32>(Shape{1, 1, 4})
        .test({1.0f, 4.0f, 1.0f, 4.0f}, f_sigmoid);
}
