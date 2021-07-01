// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "util/unary_test.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, cos_float)
{
    test::make_unary_test<TestEngine, op::Cos, element::f32>(Shape{11}).test(
        {0.f, 0.25f, -0.25f, 0.5f, -0.5f, 1.f, -1.f, 2.f, -2.f, 4.f, -4.f}, std::cos);
}

NGRAPH_TEST(${BACKEND_NAME}, cos_int)
{
    test::make_unary_test<TestEngine, op::Cos, element::i32>(Shape{5}).test(
        {1, 2, 3, 4, 5}, 
        {1, 0, -1, -1, 0});
}
